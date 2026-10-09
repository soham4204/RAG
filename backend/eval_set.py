"""Labeled questions for RAG evaluation.

Each item is one row of an evaluation dataset: the question, the fact the
answer must contain, the document that fact lives in, and the question
type. Types follow the Day 5 set: factual, numerical, comparative,
analytical, multi-hop, and unanswerable.

`expected_answer` is matched as a case-insensitive substring. When it
contains " | ", every part must appear. That is how a comparative or
multi-hop question can require two facts from the source text.

Unanswerable questions have no document id. They exist so the run can
score abstention: a good system refuses instead of inventing an answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_answer: str
    document_ids: tuple[str, ...]
    source: str
    page: int
    type: str

    @property
    def answerable(self) -> bool:
        return bool(self.document_ids)


EVAL_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id="q01",
        question="How many vacation days do employees get?",
        expected_answer="25 annual leave days",
        document_ids=("policy_leave",),
        source="employee_handbook.pdf",
        page=14,
        type="factual",
    ),
    EvalQuestion(
        id="q02",
        question="How do I get money back for my official trip?",
        expected_answer="reimbursement for approved business travel",
        document_ids=("policy_travel",),
        source="finance_policies.pdf",
        page=8,
        type="factual",
    ),
    EvalQuestion(
        id="q03",
        question="How many days each week can employees work remotely?",
        expected_answer="three days",
        document_ids=("policy_remote",),
        source="employee_handbook.pdf",
        page=22,
        type="factual",
    ),
    EvalQuestion(
        id="q04",
        question="When does health insurance coverage begin?",
        expected_answer="31st day",
        document_ids=("policy_health",),
        source="benefits_guide.pdf",
        page=3,
        type="factual",
    ),
    EvalQuestion(
        id="q05",
        question="What must employees use to sign in to email and the HR portal?",
        expected_answer="multi-factor authentication",
        document_ids=("policy_security",),
        source="it_policy.pdf",
        page=4,
        type="factual",
    ),
    EvalQuestion(
        id="q06",
        question="In which months are performance reviews held?",
        expected_answer="March and in September",
        document_ids=("policy_performance",),
        source="hr_sop.pdf",
        page=27,
        type="factual",
    ),
    EvalQuestion(
        id="q07",
        question="What year-on-year NIM change did the quarterly filing report?",
        expected_answer="17.4%",
        document_ids=("filing_q2",),
        source="q2_2026_filing.pdf",
        page=1,
        type="numerical",
    ),
    EvalQuestion(
        id="q08",
        question="What is the daily meal cap on a business trip?",
        expected_answer="75 dollars",
        document_ids=("policy_travel",),
        source="finance_policies.pdf",
        page=8,
        type="numerical",
    ),
    EvalQuestion(
        id="q09",
        question="What is the reimbursement limit without extra approval?",
        expected_answer="2000 dollars",
        document_ids=("policy_expenses",),
        source="finance_policies.pdf",
        page=15,
        type="numerical",
    ),
    EvalQuestion(
        id="q10",
        question="How many weeks of parental leave does a primary caregiver receive?",
        expected_answer="16 weeks",
        document_ids=("policy_parental",),
        source="employee_handbook.pdf",
        page=18,
        type="numerical",
    ),
    EvalQuestion(
        id="q11",
        question="What percentage of salary does the retirement plan match?",
        expected_answer="6 percent",
        document_ids=("policy_retirement",),
        source="benefits_guide.pdf",
        page=16,
        type="numerical",
    ),
    EvalQuestion(
        id="q12",
        question="How much tuition assistance can a full-time employee receive in a year?",
        expected_answer="5000 dollars",
        document_ids=("policy_tuition",),
        source="benefits_guide.pdf",
        page=9,
        type="numerical",
    ),
    EvalQuestion(
        id="q13",
        question="How does primary caregiver leave compare with secondary caregiver leave?",
        expected_answer="16 weeks | 6 weeks",
        document_ids=("policy_parental",),
        source="employee_handbook.pdf",
        page=18,
        type="comparative",
    ),
    EvalQuestion(
        id="q14",
        question="How do paid sick days compare with annual leave days?",
        expected_answer="ten paid sick days | 25 annual leave days",
        document_ids=("policy_leave",),
        source="employee_handbook.pdf",
        page=14,
        type="comparative",
    ),
    EvalQuestion(
        id="q15",
        question="What share of the valuation can a home loan cover, and are investment properties included?",
        expected_answer="80 percent | Investment properties",
        document_ids=("product_home_loan",),
        source="product_catalog.pdf",
        page=19,
        type="comparative",
    ),
    EvalQuestion(
        id="q16",
        question="Why did net interest margin increase in the quarterly filing?",
        expected_answer="higher yield on loans",
        document_ids=("filing_q2",),
        source="q2_2026_filing.pdf",
        page=1,
        type="analytical",
    ),
    EvalQuestion(
        id="q17",
        question="Why is the 17.4% NIM figure not the margin itself?",
        expected_answer="year-on-year change in the margin",
        document_ids=("filing_q2",),
        source="q2_2026_filing.pdf",
        page=1,
        type="analytical",
    ),
    EvalQuestion(
        id="q18",
        question="How many remote days are allowed, and what are the core hours?",
        expected_answer="three days | 10:00 to 16:00",
        document_ids=("policy_remote",),
        source="employee_handbook.pdf",
        page=22,
        type="multi-hop",
    ),
    EvalQuestion(
        id="q19",
        question="What is the meal cap on a business trip, and how soon must the claim be filed?",
        expected_answer="75 dollars | 30 days",
        document_ids=("policy_travel",),
        source="finance_policies.pdf",
        page=8,
        type="multi-hop",
    ),
    EvalQuestion(
        id="q20",
        question="How much tuition assistance is available, and what grade is required to claim it?",
        expected_answer="5000 dollars | grade of C",
        document_ids=("policy_tuition",),
        source="benefits_guide.pdf",
        page=9,
        type="multi-hop",
    ),
    EvalQuestion(
        id="q21",
        question="What is the company's stock price?",
        expected_answer="I don't have enough information in the provided documents to answer that.",
        document_ids=(),
        source="",
        page=0,
        type="unanswerable",
    ),
    EvalQuestion(
        id="q22",
        question="What will the company's revenue be in 2030?",
        expected_answer="I don't have enough information in the provided documents to answer that.",
        document_ids=(),
        source="",
        page=0,
        type="unanswerable",
    ),
    EvalQuestion(
        id="q23",
        question="What is on the cafeteria menu this week?",
        expected_answer="I don't have enough information in the provided documents to answer that.",
        document_ids=(),
        source="",
        page=0,
        type="unanswerable",
    ),
    EvalQuestion(
        id="q24",
        question="What is the chief executive's home address?",
        expected_answer="I don't have enough information in the provided documents to answer that.",
        document_ids=(),
        source="",
        page=0,
        type="unanswerable",
    ),
]
