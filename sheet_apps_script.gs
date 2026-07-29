/**
 * Paste this into the Apps Script editor attached to your Google Sheet:
 *   Open the sheet -> Extensions -> Apps Script -> paste this in, replacing any starter code.
 *
 * Then deploy it as a Web App:
 *   Deploy -> New deployment -> type: "Web app"
 *     - Execute as: Me
 *     - Who has access: Anyone
 *   Click Deploy, authorize it, and copy the resulting /exec URL.
 *   Put that URL into SHEET_WEBHOOK_URL in your Vercel environment variables.
 *
 * Every submission from the exam arrives here as a POST with a JSON body and
 * gets appended as one row on the "Results" sheet (created automatically on first run).
 *
 * The per-task columns are discovered from the payload itself (p1_score, p2_score, …),
 * so adding or removing hands-on tasks in api/index.py needs no change here — the
 * header row is written once, from the first submission's shape.
 */

const SHEET_NAME = "Results";

const BASE_COLUMNS = [
  ["Timestamp", "timestamp"],
  ["Name", "name"],
  ["Email", "email"],
  ["MCQ Score", "mcq_score"],
  ["MCQ Total", "mcq_total"],
  ["MCQs Correct", "mcq_correct_count"],
  ["MCQ Count", "mcq_question_count"],
  ["Hands-On Score", "practical_score"],
  ["Hands-On Total", "practical_total"],
  ["Total Score", "total_score"],
  ["Grand Total", "grand_total"],
  ["Percentage", "percentage"],
  ["Overall Feedback", "overall_feedback"],
];

function doPost(e) {
  const sheet = getOrCreateSheet_();
  const data = JSON.parse(e.postData.contents);
  const taskIds = practicalIdsIn_(data);

  ensureHeaderRow_(sheet, taskIds);

  const row = BASE_COLUMNS.map(function (col) {
    const value = data[col[1]];
    return value === undefined || value === null ? "" : value;
  });
  row[0] = data.timestamp || new Date().toISOString();

  taskIds.forEach(function (id) {
    row.push(data[id + "_score"]);
    row.push(data[id + "_feedback"] || "");
  });

  sheet.appendRow(row);

  return ContentService
    .createTextOutput(JSON.stringify({ status: "ok" }))
    .setMimeType(ContentService.MimeType.JSON);
}

/** Finds p1, p2, … in the payload and returns those ids in numeric order. */
function practicalIdsIn_(data) {
  const found = [];
  Object.keys(data).forEach(function (key) {
    const match = key.match(/^p(\d+)_score$/);
    if (match) found.push({ id: "p" + match[1], n: parseInt(match[1], 10) });
  });
  found.sort(function (a, b) { return a.n - b.n; });
  return found.map(function (x) { return x.id; });
}

function getOrCreateSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
  }
  return sheet;
}

function ensureHeaderRow_(sheet, taskIds) {
  if (sheet.getLastRow() > 0) return;

  const header = BASE_COLUMNS.map(function (col) { return col[0]; });
  taskIds.forEach(function (id, i) {
    header.push("Task " + (i + 1) + " Score");
    header.push("Task " + (i + 1) + " Feedback");
  });
  sheet.appendRow(header);
  sheet.getRange(1, 1, 1, header.length).setFontWeight("bold");
  sheet.setFrozenRows(1);
}
