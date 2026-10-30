import os
import re
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

SQL_FILE = PROJECT_ROOT / "sql" / "sql_business_questions.sql"

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "reports" / "sql_question_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL)


def load_questions(sql_file):
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    content = sql_file.read_text(encoding="utf-8")

    pattern = re.compile(r"(?im)^\s*--\s*(Q\d+)\s*:")
    matches = list(pattern.finditer(content))

    questions = []

    for index, match in enumerate(matches):
        question_id = match.group(1).upper()

        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        section = content[start:end]

        # Find the actual SQL start, ignoring question descriptions.
        lines = section.splitlines()

        sql_start = None

        for line_index, line in enumerate(lines):
            if re.match(r"^\s*(SELECT|WITH)\b", line, re.IGNORECASE):
                sql_start = line_index
                break

        if sql_start is None:
            raise ValueError(
                f"No SELECT/WITH query found for {question_id}"
            )

        query = "\n".join(lines[sql_start:]).strip()
        query = query.rstrip(";").strip()

        if query:
            questions.append((question_id, query))

    return questions


def run_questions():
    questions = load_questions(SQL_FILE)

    if not questions:
        raise ValueError("No SQL questions found.")

    question_ids = [qid for qid, _ in questions]

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs found.")

    summary = []

    print("\n" + "=" * 60)
    print("NETFLIX CUSTOMER INTELLIGENCE")
    print("AUTOMATED SQL BUSINESS QUESTION RUNNER")
    print("=" * 60)

    print(f"\nQuestions detected: {len(questions)}")

    for question_id, query in questions:
        print("\n" + "-" * 60)
        print(f"Executing {question_id}...")

        start_time = time.perf_counter()

        try:
            # Only read-only SELECT/WITH queries are allowed.
            if not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
                raise ValueError("Only SELECT/WITH queries are allowed.")

            with engine.connect() as connection:
                result = connection.execute(text(query))

                if not result.returns_rows:
                    raise ValueError("Query did not return rows.")

                columns = list(result.keys())
                rows = result.fetchall()

            output_file = OUTPUT_DIR / f"{question_id}.csv"

            with output_file.open(
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=columns
                )
                writer.writeheader()

                for row in rows:
                    writer.writerow({
                        column: row._mapping[column]
                        for column in columns
                    })

            status = "PASS"
            error_message = ""

            print(f"{question_id}: PASS | Rows: {len(rows)}")

        except Exception as error:
            status = "FAIL"
            error_message = str(error)
            output_file = None

            print(f"{question_id}: FAIL")
            print(f"Error: {error_message}")

        duration = round(time.perf_counter() - start_time, 3)

        summary.append({
            "question_id": question_id,
            "status": status,
            "result_rows": len(rows) if status == "PASS" else 0,
            "execution_seconds": duration,
            "result_file": (
                str(output_file.relative_to(PROJECT_ROOT))
                if output_file else ""
            ),
            "error": error_message
        })

    summary_file = OUTPUT_DIR / "execution_summary.csv"

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        fields = [
            "question_id",
            "status",
            "result_rows",
            "execution_seconds",
            "result_file",
            "error"
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    passed = sum(x["status"] == "PASS" for x in summary)
    failed = len(summary) - passed

    print("\n" + "=" * 60)
    print("FINAL EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(summary)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"\nReport: {summary_file}")
    print(f"Results folder: {OUTPUT_DIR}")

    engine.dispose()


if __name__ == "__main__":
    run_questions()
