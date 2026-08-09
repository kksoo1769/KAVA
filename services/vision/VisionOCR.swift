"""
xcrun swiftc -O \
  -target arm64-apple-macosx26.0 \
  services/vision/VisionOCR.swift \
  -o services/vision/vision-ocr \
  -framework Vision \
  -framework ImageIO \
  -framework DataDetection

services/vision/vision-ocr --mode hybrid \
  analysis/vlm_comparison/heldout_kdtc/table_22.jpg | jq \
  '.document.blocks[] | select(.type == "table") | .table.rows'
"""

import Foundation
import Vision
import ImageIO
import DataDetection

struct OCRRow: Encodable {
    let idx: Int
    let text: String
    let bbox: [Double]
    let confidenceScore: Float

    enum CodingKeys: String, CodingKey {
        case idx, text, bbox
        case confidenceScore = "confidence_score"
    }
}

enum OCRMode: String {
    case legacy
    case document
    case numericAccurate = "numeric-accurate"
    case numericFast = "numeric-fast"
    case hybrid
}

struct CLIOptions {
    let mode: OCRMode
    let path: String
}

struct LineOutput: Encodable {
    let id: String
    let text: String
    let bbox: [Double]
    let confidence: Float
    let source: String
    let candidates: [String]
    let isTitle: Bool?
    let shouldWrap: Bool?
    let languages: [String]
}

struct DetectedDataOutput: Encodable {
    let kind: String
    let value: String
    let bbox: [Double]
}

struct BarcodeOutput: Encodable {
    let payload: String?
    let symbology: String
    let confidence: Float
    let bbox: [Double]
    let polygon: [[Double]]
}

struct CellOutput: Encodable {
    let id: String
    let rowRange: [Int]
    let columnRange: [Int]
    var text: String
    let bbox: [Double]
    let polygon: [[Double]]
    var source: String
    var confidence: Float?
}

struct TableOutput: Encodable {
    let rowCount: Int
    let columnCount: Int
    var rows: [[CellOutput]]
}

struct ListItemOutput: Encodable {
    let marker: String
    let text: String
    let bbox: [Double]
    let polygon: [[Double]]
}

struct ListOutput: Encodable {
    let items: [ListItemOutput]
}

struct BlockOutput: Encodable {
    let id: String
    let type: String
    let text: String?
    let bbox: [Double]
    let polygon: [[Double]]
    var table: TableOutput?
    let list: ListOutput?
}

struct DocumentPayload: Encodable {
    let title: String?
    let transcript: String
    var blocks: [BlockOutput]
    let lines: [LineOutput]
    let detectedData: [DetectedDataOutput]
    let barcodes: [BarcodeOutput]
}

struct DocumentResponse: Encodable {
    let schemaVersion: Int
    let engine: String
    let mode: String
    let coordinateOrigin: String
    var document: DocumentPayload
}

struct PassResponse: Encodable {
    let schemaVersion: Int
    let engine: String
    let mode: String
    let coordinateOrigin: String
    let lines: [LineOutput]
}

struct RecognizedItem {
    var text: String
    var confidence: Float
    let bbox: CGRect
    let candidates: [String]
}

enum CLIError: Error, CustomStringConvertible {
    case usage
    case invalidMode(String)
    case unreadableImage(String)
    case noCGImage(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: vision-ocr [--mode legacy|document|numeric-accurate|numeric-fast|hybrid] <image-path>"
        case .invalidMode(let mode):
            return "invalid mode: \(mode)"
        case .unreadableImage(let path):
            return "cannot open image: \(path)"
        case .noCGImage(let path):
            return "cannot decode image: \(path)"
        }
    }
}

func imageOrientation(
    from source: CGImageSource
) -> CGImagePropertyOrientation {
    guard
        let properties = CGImageSourceCopyPropertiesAtIndex(
            source,
            0,
            nil
        ) as? [CFString: Any],
        let raw = (
            properties[kCGImagePropertyOrientation] as? NSNumber
        )?.uint32Value,
        let orientation = CGImagePropertyOrientation(rawValue: raw)
    else {
        return .up
    }

    return orientation
}

func makeTextRequest(
    level: VNRequestTextRecognitionLevel,
    languages: [String],
    usesLanguageCorrection: Bool,
    minimumTextHeight: Float?
) -> VNRecognizeTextRequest {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = level
    request.recognitionLanguages = languages
    request.usesLanguageCorrection = usesLanguageCorrection

    // 0보다 큰 값은 작은 글자를 추가로 읽는 옵션이 아니라,
    // 해당 높이보다 작은 글자를 제외하는 필터다.
    if let minimumTextHeight {
        request.minimumTextHeight = minimumTextHeight
    }

    return request
}

func performTextRequest(
    _ request: VNRecognizeTextRequest,
    image: CGImage,
    orientation: CGImagePropertyOrientation
) throws -> [RecognizedItem] {
    let handler = VNImageRequestHandler(
        cgImage: image,
        orientation: orientation,
        options: [:]
    )

    try handler.perform([request])

    return (request.results ?? []).compactMap { observation in
        let candidates = observation.topCandidates(5)
        guard let candidate = candidates.first else {
            return nil
        }

        return RecognizedItem(
            text: candidate.string,
            confidence: candidate.confidence,
            bbox: observation.boundingBox,
            candidates: candidates.map(\.string)
        )
    }
}

let numericCharacters = CharacterSet(
    charactersIn: "0123456789.,%+-−₩$€£¥()/:' "
).union(.whitespacesAndNewlines)

func isStrictlyNumeric(_ text: String) -> Bool {
    let scalars = text.unicodeScalars
    let hasDigit = scalars.contains {
        CharacterSet.decimalDigits.contains($0)
    }

    return hasDigit && scalars.allSatisfy {
        numericCharacters.contains($0)
    }
}

func digitCount(_ text: String) -> Int {
    text.unicodeScalars.reduce(into: 0) { count, scalar in
        if CharacterSet.decimalDigits.contains(scalar) {
            count += 1
        }
    }
}

func overlapRatio(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
    let intersection = lhs.intersection(rhs)

    guard !intersection.isNull, !intersection.isEmpty else {
        return 0
    }

    let intersectionArea = intersection.width * intersection.height
    let smallerArea = min(
        lhs.width * lhs.height,
        rhs.width * rhs.height
    )

    guard smallerArea > 0 else {
        return 0
    }

    return intersectionArea / smallerArea
}

func verticalOverlapRatio(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
    let overlap = min(lhs.maxY, rhs.maxY) - max(lhs.minY, rhs.minY)

    guard overlap > 0 else {
        return 0
    }

    return overlap / min(lhs.height, rhs.height)
}

func replaceNumericCandidates(
    in baseline: [RecognizedItem],
    with numericPass: [RecognizedItem]
) -> [RecognizedItem] {
    baseline.map { original in
        guard isStrictlyNumeric(original.text) else {
            return original
        }

        let matches = numericPass.filter {
            isStrictlyNumeric($0.text)
                && overlapRatio(original.bbox, $0.bbox) >= 0.55
        }

        guard
            let replacement = matches.max(by: {
                overlapRatio(original.bbox, $0.bbox)
                    < overlapRatio(original.bbox, $1.bbox)
            }),
            digitCount(original.text) == digitCount(replacement.text)
        else {
            return original
        }

        return RecognizedItem(
            text: replacement.text,
            confidence: replacement.confidence,
            bbox: original.bbox,
            candidates: replacement.candidates
        )
    }
}

func missingNumericColumnItems(
    baseline: [RecognizedItem],
    fastPass: [RecognizedItem]
) -> [RecognizedItem] {
    let candidates = fastPass.filter { candidate in
        guard isStrictlyNumeric(candidate.text) else {
            return false
        }

        let duplicatesExistingItem = baseline.contains {
            overlapRatio(candidate.bbox, $0.bbox) >= 0.45
        }
        guard !duplicatesExistingItem else {
            return false
        }

        // 표의 다른 셀과 같은 행에 있는 숫자만 후보로 둔다.
        return baseline.contains {
            verticalOverlapRatio(candidate.bbox, $0.bbox) >= 0.5
        }
    }
    .sorted { $0.bbox.midX < $1.bbox.midX }

    var clusters: [[RecognizedItem]] = []

    for candidate in candidates {
        if let index = clusters.indices.first(where: { index in
            let center = clusters[index]
                .map(\.bbox.midX)
                .reduce(0, +) / CGFloat(clusters[index].count)
            return abs(center - candidate.bbox.midX) <= 0.035
        }) {
            clusters[index].append(candidate)
        } else {
            clusters.append([candidate])
        }
    }

    // 우연히 숫자로 오인한 글자를 추가하지 않도록, 같은 x 위치에서
    // 최소 3개 행이 반복될 때만 누락된 숫자 열로 판단한다.
    return clusters
        .filter { cluster in
            let rowCenters = cluster.map { $0.bbox.midY }.sorted()
            let uniqueRows = rowCenters.reduce(into: [CGFloat]()) {
                rows, center in
                if rows.last.map({ abs($0 - center) > 0.01 }) ?? true {
                    rows.append(center)
                }
            }
            return uniqueRows.count >= 3
        }
        .flatMap { $0 }
}

func isIntegerText(_ text: String) -> Bool {
    !text.isEmpty && text.unicodeScalars.allSatisfy {
        CharacterSet.decimalDigits.contains($0)
    }
}

func isDecimalZeroText(_ text: String) -> Bool {
    guard text.hasSuffix(".0") else {
        return false
    }

    return isIntegerText(String(text.dropLast(2)))
}

func restoreRowDecimalPoints(
    in items: [RecognizedItem]
) -> [RecognizedItem] {
    items.map { item in
        guard
            isIntegerText(item.text),
            item.text.count >= 2,
            item.text.last == "0"
        else {
            return item
        }

        let decimalZeroPeers = items.filter {
            overlapRatio(item.bbox, $0.bbox) < 0.45
                && verticalOverlapRatio(item.bbox, $0.bbox) >= 0.5
                && isDecimalZeroText($0.text)
        }

        // 한두 개의 우연한 소수에 끌려가지 않고, 같은 행의 형식이
        // 충분히 반복되는 숫자 표에서만 누락된 소수점을 복원한다.
        guard decimalZeroPeers.count >= 3 else {
            return item
        }

        var corrected = item.text
        corrected.insert(".", at: corrected.index(before: corrected.endIndex))

        return RecognizedItem(
            text: corrected,
            confidence: item.confidence,
            bbox: item.bbox,
            candidates: item.candidates
        )
    }
}

func rounded(_ value: CGFloat) -> Double {
    (Double(value) * 1_000_000).rounded() / 1_000_000
}

func topLeftBBox(_ rect: CGRect) -> [Double] {
    [
        rounded(rect.minX),
        rounded(1.0 - rect.maxY),
        rounded(rect.maxX),
        rounded(1.0 - rect.minY),
    ]
}

func topLeftPolygon(_ region: NormalizedRegion) -> [[Double]] {
    region.points.map { point in
        [rounded(point.x), rounded(1.0 - point.y)]
    }
}

func coverageRatio(_ item: CGRect, _ container: CGRect) -> CGFloat {
    let intersection = item.intersection(container)
    guard !intersection.isNull, !intersection.isEmpty else {
        return 0
    }

    let itemArea = item.width * item.height
    guard itemArea > 0 else {
        return 0
    }

    return intersection.width * intersection.height / itemArea
}

func lineCenterIsInside(_ line: LineOutput, bbox: [Double]) -> Bool {
    guard line.bbox.count == 4, bbox.count == 4 else {
        return false
    }

    let x = (line.bbox[0] + line.bbox[2]) / 2
    let y = (line.bbox[1] + line.bbox[3]) / 2
    return bbox[0] <= x && x <= bbox[2]
        && bbox[1] <= y && y <= bbox[3]
}

func makeLineOutputs(
    _ items: [RecognizedItem],
    source: String
) -> [LineOutput] {
    items.enumerated().map { index, item in
        LineOutput(
            id: "line-\(index)",
            text: item.text,
            bbox: topLeftBBox(item.bbox),
            confidence: item.confidence,
            source: source,
            candidates: item.candidates,
            isTitle: nil,
            shouldWrap: nil,
            languages: []
        )
    }
}

func makeDocumentLineOutput(
    _ line: RecognizedTextObservation,
    index: Int
) -> LineOutput {
    LineOutput(
        id: "line-\(index)",
        text: line.transcript,
        bbox: topLeftBBox(line.boundingBox.cgRect),
        confidence: line.confidence,
        source: "document",
        candidates: line.topCandidates(5).map(\.string),
        isTitle: line.isTitle,
        shouldWrap: line.shouldWrapToNextLine,
        languages: line.recognitionLanguages.map(\.minimalIdentifier)
    )
}

func detectedDataValue(
    _ detected: DocumentObservation.Container.DataDetectorMatch,
    transcript: String
) -> DetectedDataOutput {
    let fallbackValue: String = {
        guard let range = detected.match.range else {
            return ""
        }
        return String(transcript[range])
    }()

    let kind: String
    let value: String

    switch detected.match.details {
    case .link(let details):
        kind = "link"
        value = details.url.absoluteString
    case .emailAddress(let details):
        kind = "email_address"
        value = details.emailAddress
    case .phoneNumber(let details):
        kind = "phone_number"
        value = details.phoneNumber
    case .postalAddress(let details):
        kind = "postal_address"
        value = details.fullAddress
    case .calendarEvent(let details):
        kind = "calendar_event"
        value = details.startDate.map {
            ISO8601DateFormatter().string(from: $0)
        } ?? fallbackValue
    case .moneyAmount(let details):
        kind = "money_amount"
        value = "\(details.amount) \(details.currency.identifier)"
    case .flightNumber(let details):
        kind = "flight_number"
        value = "\(details.airlineCode)\(details.flightNumber)"
    case .shipmentTrackingNumber(let details):
        kind = "shipment_tracking_number"
        value = details.trackingNumber
    case .measurement(let details):
        kind = "measurement"
        value = fallbackValue.isEmpty ? String(details.value) : fallbackValue
    case .paymentIdentifier(let details):
        kind = "payment_identifier"
        value = details.identifier
    @unknown default:
        kind = "unknown"
        value = fallbackValue
    }

    return DetectedDataOutput(
        kind: kind,
        value: value,
        bbox: topLeftBBox(detected.boundingRegion.boundingBox.cgRect)
    )
}

func averageConfidence(
    _ lines: [RecognizedTextObservation]
) -> Float? {
    guard !lines.isEmpty else {
        return nil
    }
    return lines.map(\.confidence).reduce(0, +) / Float(lines.count)
}

func performDocumentRequest(
    url: URL,
    orientation: CGImagePropertyOrientation,
    minimumTextHeight: Float?
) async throws -> DocumentResponse {
    var request = RecognizeDocumentsRequest()
    var options = request.textRecognitionOptions
    options.recognitionLanguages = [
        Locale.Language(identifier: "ko-KR"),
        Locale.Language(identifier: "en-US"),
    ]
    options.automaticallyDetectLanguage = false
    options.useLanguageCorrection = true
    options.maximumCandidateCount = 5
    if let minimumTextHeight {
        options.minimumTextHeightFraction = minimumTextHeight
    }
    if let customWords = ProcessInfo.processInfo.environment[
        "VISION_CUSTOM_WORDS"
    ] {
        options.customWords = customWords
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
    request.textRecognitionOptions = options

    let observations = try await request.perform(
        on: url,
        orientation: orientation
    )
    guard let observation = observations.first else {
        return DocumentResponse(
            schemaVersion: 2,
            engine: "apple-vision-document",
            mode: OCRMode.document.rawValue,
            coordinateOrigin: "top-left",
            document: DocumentPayload(
                title: nil,
                transcript: "",
                blocks: [],
                lines: [],
                detectedData: [],
                barcodes: []
            )
        )
    }

    let document = observation.document
    let documentLines = document.text.lines.enumerated().map {
        makeDocumentLineOutput($0.element, index: $0.offset)
    }
    var blocks: [(output: BlockOutput, rect: CGRect)] = []
    var coveredRegions: [CGRect] = []

    if let title = document.title {
        let rect = title.boundingRegion.boundingBox.cgRect
        blocks.append((
            BlockOutput(
                id: "title-0",
                type: "title",
                text: title.transcript,
                bbox: topLeftBBox(rect),
                polygon: topLeftPolygon(title.boundingRegion),
                table: nil,
                list: nil
            ),
            rect
        ))
        coveredRegions.append(rect)
    }

    for (tableIndex, table) in document.tables.enumerated() {
        let rect = table.boundingRegion.boundingBox.cgRect
        let rows = table.rows.enumerated().map { rowIndex, row in
            row.enumerated().map { columnIndex, cell in
                let content = cell.content.text
                let cellRect = cell.content.boundingRegion.boundingBox.cgRect
                return CellOutput(
                    id: "table-\(tableIndex)-cell-\(rowIndex)-\(columnIndex)",
                    rowRange: [
                        cell.rowRange.lowerBound,
                        cell.rowRange.upperBound,
                    ],
                    columnRange: [
                        cell.columnRange.lowerBound,
                        cell.columnRange.upperBound,
                    ],
                    text: content.transcript,
                    bbox: topLeftBBox(cellRect),
                    polygon: topLeftPolygon(cell.content.boundingRegion),
                    source: content.transcript.isEmpty
                        ? "unrecognized" : "document",
                    confidence: averageConfidence(content.lines)
                )
            }
        }
        blocks.append((
            BlockOutput(
                id: "table-\(tableIndex)",
                type: "table",
                text: nil,
                bbox: topLeftBBox(rect),
                polygon: topLeftPolygon(table.boundingRegion),
                table: TableOutput(
                    rowCount: table.rows.count,
                    columnCount: table.columns.count,
                    rows: rows
                ),
                list: nil
            ),
            rect
        ))
        coveredRegions.append(rect)
    }

    for (listIndex, list) in document.lists.enumerated() {
        let rect = list.boundingRegion.boundingBox.cgRect
        let items = list.items.map { item in
            let itemRect = item.content.boundingRegion.boundingBox.cgRect
            return ListItemOutput(
                marker: item.markerString,
                text: item.itemString,
                bbox: topLeftBBox(itemRect),
                polygon: topLeftPolygon(item.content.boundingRegion)
            )
        }
        blocks.append((
            BlockOutput(
                id: "list-\(listIndex)",
                type: "list",
                text: nil,
                bbox: topLeftBBox(rect),
                polygon: topLeftPolygon(list.boundingRegion),
                table: nil,
                list: ListOutput(items: items)
            ),
            rect
        ))
        coveredRegions.append(rect)
    }

    for (paragraphIndex, paragraph) in document.paragraphs.enumerated() {
        let rect = paragraph.boundingRegion.boundingBox.cgRect
        let isCovered = coveredRegions.contains {
            coverageRatio(rect, $0) >= 0.8
        }
        guard !isCovered else {
            continue
        }

        blocks.append((
            BlockOutput(
                id: "paragraph-\(paragraphIndex)",
                type: "paragraph",
                text: paragraph.transcript,
                bbox: topLeftBBox(rect),
                polygon: topLeftPolygon(paragraph.boundingRegion),
                table: nil,
                list: nil
            ),
            rect
        ))
    }

    blocks.sort {
        let lhsTop = 1.0 - $0.rect.maxY
        let rhsTop = 1.0 - $1.rect.maxY
        if abs(lhsTop - rhsTop) <= 0.005 {
            return $0.rect.minX < $1.rect.minX
        }
        return lhsTop < rhsTop
    }

    let detectedData = document.text.detectedData.map {
        detectedDataValue($0, transcript: document.text.transcript)
    }
    let barcodes = document.barcodes.map { barcode in
        BarcodeOutput(
            payload: barcode.payloadString,
            symbology: String(describing: barcode.symbology),
            confidence: barcode.confidence,
            bbox: topLeftBBox(barcode.boundingBox.cgRect),
            polygon: topLeftPolygon(barcode.boundingRegion)
        )
    }

    return DocumentResponse(
        schemaVersion: 2,
        engine: "apple-vision-document",
        mode: OCRMode.document.rawValue,
        coordinateOrigin: "top-left",
        document: DocumentPayload(
            title: document.title?.transcript,
            transcript: document.text.transcript,
            blocks: blocks.map(\.output),
            lines: documentLines,
            detectedData: detectedData,
            barcodes: barcodes
        )
    )
}

func mergeHybrid(
    documentResponse: DocumentResponse,
    accurateItems: [RecognizedItem],
    fastItems: [RecognizedItem]
) -> DocumentResponse {
    var response = documentResponse
    let accurateLines = makeLineOutputs(
        accurateItems,
        source: "numeric-accurate"
    )
    let fastLines = makeLineOutputs(
        fastItems,
        source: "numeric-fast"
    )
    let documentLines = response.document.lines

    for blockIndex in response.document.blocks.indices {
        guard var table = response.document.blocks[blockIndex].table else {
            continue
        }

        for rowIndex in table.rows.indices {
            for cellIndex in table.rows[rowIndex].indices {
                var cell = table.rows[rowIndex][cellIndex]

                if cell.text.isEmpty {
                    let matches = documentLines.filter {
                        lineCenterIsInside($0, bbox: cell.bbox)
                    }
                    if !matches.isEmpty {
                        cell.text = matches.map(\.text).joined(separator: " ")
                        cell.source = "document-line"
                        cell.confidence = matches.map(\.confidence).max()
                    }
                }

                let accurateMatches = accurateLines.filter {
                    isStrictlyNumeric($0.text)
                        && lineCenterIsInside($0, bbox: cell.bbox)
                }
                if
                    isStrictlyNumeric(cell.text),
                    let replacement = accurateMatches.first,
                    digitCount(cell.text) == digitCount(replacement.text)
                {
                    cell.text = replacement.text
                    cell.source = "numeric-accurate"
                    cell.confidence = replacement.confidence
                }

                if cell.text.isEmpty {
                    let fastMatches = fastLines.filter {
                        isStrictlyNumeric($0.text)
                            && lineCenterIsInside($0, bbox: cell.bbox)
                    }
                    if let replacement = fastMatches.first {
                        cell.text = replacement.text
                        cell.source = "numeric-fast"
                        cell.confidence = replacement.confidence
                    }
                }

                table.rows[rowIndex][cellIndex] = cell
            }

            let decimalPeerCount = table.rows[rowIndex].filter {
                isDecimalZeroText($0.text)
            }.count
            if decimalPeerCount >= 3 {
                for cellIndex in table.rows[rowIndex].indices {
                    var cell = table.rows[rowIndex][cellIndex]
                    if
                        isIntegerText(cell.text),
                        cell.text.count >= 2,
                        cell.text.last == "0"
                    {
                        cell.text.insert(
                            ".",
                            at: cell.text.index(before: cell.text.endIndex)
                        )
                        cell.source = "row-consensus"
                        table.rows[rowIndex][cellIndex] = cell
                    }
                }
            }
        }

        response.document.blocks[blockIndex].table = table
    }

    return DocumentResponse(
        schemaVersion: response.schemaVersion,
        engine: "apple-vision-hybrid",
        mode: OCRMode.hybrid.rawValue,
        coordinateOrigin: response.coordinateOrigin,
        document: response.document
    )
}

func parseCLI() throws -> CLIOptions {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments.count == 1 {
        return CLIOptions(mode: .legacy, path: arguments[0])
    }

    guard arguments.count == 3, arguments[0] == "--mode" else {
        throw CLIError.usage
    }
    guard let mode = OCRMode(rawValue: arguments[1]) else {
        throw CLIError.invalidMode(arguments[1])
    }
    return CLIOptions(mode: mode, path: arguments[2])
}

func loadImage(
    path: String
) throws -> (
    url: URL,
    image: CGImage,
    orientation: CGImagePropertyOrientation
) {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw CLIError.unreadableImage(path)
    }
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw CLIError.noCGImage(path)
    }
    return (url, image, imageOrientation(from: source))
}

func sortReadingOrder(_ items: inout [RecognizedItem]) {
    items.sort {
        let lhsTop = 1.0 - $0.bbox.maxY
        let rhsTop = 1.0 - $1.bbox.maxY
        if abs(lhsTop - rhsTop) <= 0.005 {
            return $0.bbox.minX < $1.bbox.minX
        }
        return lhsTop < rhsTop
    }
}

func performNumericPass(
    mode: OCRMode,
    image: CGImage,
    orientation: CGImagePropertyOrientation,
    minimumTextHeight: Float?
) throws -> [RecognizedItem] {
    let isFast = mode == .numericFast
    let request = makeTextRequest(
        level: isFast ? .fast : .accurate,
        languages: ["en-US"],
        usesLanguageCorrection: !isFast,
        minimumTextHeight: minimumTextHeight
    )
    var items = try performTextRequest(
        request,
        image: image,
        orientation: orientation
    )
    sortReadingOrder(&items)
    return items
}

func performLegacy(
    image: CGImage,
    orientation: CGImagePropertyOrientation,
    minimumTextHeight: Float?,
    environment: [String: String]
) throws -> [OCRRow] {
    let baselineRequest = makeTextRequest(
        level: .accurate,
        languages: ["ko-KR", "en-US"],
        usesLanguageCorrection: true,
        minimumTextHeight: minimumTextHeight
    )
    var recognized = try performTextRequest(
        baselineRequest,
        image: image,
        orientation: orientation
    )

    if environment["VISION_NUMERIC_MULTIPASS"] != "0" {
        if let numericResults = try? performNumericPass(
            mode: .numericAccurate,
            image: image,
            orientation: orientation,
            minimumTextHeight: minimumTextHeight
        ) {
            recognized = replaceNumericCandidates(
                in: recognized,
                with: numericResults
            )
        }
        if let fastResults = try? performNumericPass(
            mode: .numericFast,
            image: image,
            orientation: orientation,
            minimumTextHeight: minimumTextHeight
        ) {
            recognized.append(contentsOf: missingNumericColumnItems(
                baseline: recognized,
                fastPass: fastResults
            ))
        }
        recognized = restoreRowDecimalPoints(in: recognized)
    }
    sortReadingOrder(&recognized)

    return recognized.enumerated().map { idx, item in
        OCRRow(
            idx: idx,
            text: item.text,
            bbox: topLeftBBox(item.bbox).map {
                ($0 * 1000).rounded() / 1000
            },
            confidenceScore:
                (item.confidence * 100).rounded() / 100
        )
    }
}

func writeJSON<T: Encodable>(_ value: T, snakeCase: Bool) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes, .sortedKeys]
    if snakeCase {
        encoder.keyEncodingStrategy = .convertToSnakeCase
    }
    let data = try encoder.encode(value)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

func run() async throws {
    let options = try parseCLI()
    let loaded = try loadImage(path: options.path)
    let environment = ProcessInfo.processInfo.environment
    let minimumTextHeight = environment["VISION_MIN_TEXT_HEIGHT"]
        .flatMap(Float.init)

    switch options.mode {
    case .legacy:
        let rows = try performLegacy(
            image: loaded.image,
            orientation: loaded.orientation,
            minimumTextHeight: minimumTextHeight,
            environment: environment
        )
        try writeJSON(rows, snakeCase: false)

    case .numericAccurate, .numericFast:
        let items = try performNumericPass(
            mode: options.mode,
            image: loaded.image,
            orientation: loaded.orientation,
            minimumTextHeight: minimumTextHeight
        )
        let response = PassResponse(
            schemaVersion: 2,
            engine: "apple-vision-text",
            mode: options.mode.rawValue,
            coordinateOrigin: "top-left",
            lines: makeLineOutputs(items, source: options.mode.rawValue)
        )
        try writeJSON(response, snakeCase: true)

    case .document:
        let response = try await performDocumentRequest(
            url: loaded.url,
            orientation: loaded.orientation,
            minimumTextHeight: minimumTextHeight
        )
        try writeJSON(response, snakeCase: true)

    case .hybrid:
        let documentResponse = try await performDocumentRequest(
            url: loaded.url,
            orientation: loaded.orientation,
            minimumTextHeight: minimumTextHeight
        )
        let accurateItems = try performNumericPass(
            mode: .numericAccurate,
            image: loaded.image,
            orientation: loaded.orientation,
            minimumTextHeight: minimumTextHeight
        )
        let fastItems = try performNumericPass(
            mode: .numericFast,
            image: loaded.image,
            orientation: loaded.orientation,
            minimumTextHeight: minimumTextHeight
        )
        let response = mergeHybrid(
            documentResponse: documentResponse,
            accurateItems: accurateItems,
            fastItems: fastItems
        )
        try writeJSON(response, snakeCase: true)
    }
}

do {
    try await run()
} catch {
    FileHandle.standardError.write(
        Data("vision-ocr: \(error)\n".utf8)
    )
    exit(1)
}
