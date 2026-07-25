/**
 * Paste this into the Apps Script editor attached to your Google Sheet:
 *   Open the sheet -> Extensions -> Apps Script -> paste this in, replacing any starter code.
 *
 * Then deploy it as a Web App:
 *   Deploy -> New deployment -> type: "Web app"
 *     - Execute as: Me
 *     - Who has access: Anyone
 *   Click Deploy, authorize it, and copy the resulting /exec URL.
 *   Put that URL into SHEET_WEBHOOK_URL in your .env file.
 *
 * Every submission from the app arrives here as a POST with a JSON body and
 * gets appended as one row on the "Results" sheet (created automatically on first run).
 */

const SHEET_NAME = "Results";
const NUM_TASKS = 17;

function doPost(e) {
  const sheet = getOrCreateSheet_();
  const data = JSON.parse(e.postData.contents);

  ensureHeaderRow_(sheet);

  const row = [
    data.timestamp || new Date().toISOString(),
    data.name || "",
    data.email || "",
    data.total_score,
    data.max_score,
    data.percentage,
    data.overall_feedback || "",
  ];

  for (let i = 1; i <= NUM_TASKS; i++) {
    row.push(data["task" + i + "_score"]);
    row.push(data["task" + i + "_feedback"] || "");
  }

  sheet.appendRow(row);

  return ContentService
    .createTextOutput(JSON.stringify({ status: "ok" }))
    .setMimeType(ContentService.MimeType.JSON);
}

function getOrCreateSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
  }
  return sheet;
}

function ensureHeaderRow_(sheet) {
  if (sheet.getLastRow() > 0) return;

  const header = ["Timestamp", "Name", "Email", "Total Score", "Max Score", "Percentage", "Overall Feedback"];
  for (let i = 1; i <= NUM_TASKS; i++) {
    header.push("Task " + i + " Score");
    header.push("Task " + i + " Feedback");
  }
  sheet.appendRow(header);
  sheet.getRange(1, 1, 1, header.length).setFontWeight("bold");
  sheet.setFrozenRows(1);
}
