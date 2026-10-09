"""Sample company documents for the two demos.

A real indexing pipeline would parse PDFs and other files into text.
Here each item stands in for one parsed source document, with the
metadata a vector database would store beside the vector: title,
section, source file, and page.

The facts are written so the course examples have a clear home:

- "vacation days" should find the annual leave policy, which never
  uses the word "vacation" (semantic match, not a keyword match).
- "money back for an official trip" should find travel reimbursement.
- "student loan" expenses should find the education loan product.
- "17.4% NIM" should find the quarterly filing (an exact-number case).
- "stock price" appears nowhere, so retrieval has nothing relevant.
"""

DOCUMENTS: list[dict] = [
    {
        "id": "policy_leave",
        "title": "Annual Leave Policy",
        "section": "Leave Policy",
        "source": "employee_handbook.pdf",
        "page": 14,
        "text": """
Employees receive 25 annual leave days per calendar year. Northwind
Financial gives full-time employees paid time away from work so they can
rest and handle personal matters. This section covers public holidays,
sick leave, and annual leave. They are separate balances and are
requested in different ways.

Public holidays follow the regional calendar published by Human Resources
each January. A holiday that falls on a weekend is not automatically
moved. Employees who must work on a public holiday receive a substitute
day off within the same quarter.

Sick leave is for illness or injury. Employees receive ten paid sick days
each calendar year. A manager may ask for a medical note when sick leave
lasts more than three consecutive working days. Sick leave does not carry
into the next year and is not paid out when someone leaves the company.

The annual leave balance starts on 1 January for anyone who has finished
probation. New hires
accrue leave monthly during probation and may not take annual leave until
probation ends. Up to 5 unused annual leave days may be carried into the
next year. Days above that carry-over limit expire on 31 March.

Requests for annual leave should be submitted in the HR portal at least
ten working days in advance. Managers approve or decline based on team
coverage. Leave already approved can be cancelled only when the manager
agrees. Annual leave is paid at the employee's normal base salary and
does not include overtime or bonuses.
""".strip(),
    },
    {
        "id": "policy_remote",
        "title": "Remote Work Policy",
        "section": "Workplace",
        "source": "employee_handbook.pdf",
        "page": 22,
        "text": """
Employees may work remotely three days each week. Northwind Financial
allows hybrid work for roles that do not require a daily presence in a
branch or operations centre. The other two days are spent in the office.
The employee's manager sets which weekdays are in-office days.

Core hours are 10:00 to 16:00 in the employee's local time zone. During
core hours the employee is expected to be reachable on chat and to attend
scheduled meetings. Work outside core hours is allowed when the role
requires it, but it does not reduce the in-office requirement.

Fully remote arrangements are an exception. They need written approval
from the manager and the head of the department, and they are reviewed
every six months. Client-facing branch staff, treasury settlement staff,
and data-centre technicians are not eligible for remote work.

Equipment used at home must be the company laptop. Household computers
must not be used for customer data. A home workspace should be private
enough that calls cannot be overheard. The company does not reimburse
home internet or electricity.
""".strip(),
    },
    {
        "id": "policy_travel",
        "title": "Business Travel Reimbursement",
        "section": "Travel",
        "source": "finance_policies.pdf",
        "page": 8,
        "text": """
Employees can request reimbursement for approved business travel expenses.
Travel is approved before the trip in the expense system. A trip that was
not approved in advance is not reimbursed, except when a manager confirms
it was an emergency customer visit.

Covered costs are transport to the destination, a standard hotel room,
and meals while away from the home city. Meals are capped at 75 dollars
per person per day. Alcohol, minibar charges, and entertainment are not
covered. Receipts are required for every item above 25 dollars.

The employee submits the claim within 30 days of returning. Finance pays
an approved claim in the next payroll cycle. Claims submitted after 30
days are rejected unless the chief financial officer grants an exception.
Personal travel added onto a business trip is paid by the employee. Only
the business portion of the fare can be claimed.
""".strip(),
    },
    {
        "id": "policy_health",
        "title": "Health Insurance",
        "section": "Benefits",
        "source": "benefits_guide.pdf",
        "page": 3,
        "text": """
The company provides health insurance to full-time employees. Coverage
begins on the 31st day of employment. Part-time staff below 30 hours a
week and contractors are not enrolled in the company plan.

The plan covers the employee. A spouse and dependent children can be added
during onboarding or in the annual enrollment window in November. Adding
a dependent after a qualifying life event is allowed within 30 days of
that event. Premiums for dependents are deducted from salary.

Dental and vision are included in the same plan. Elective cosmetic
procedures are excluded. Employees can find the insurer's member number
and the claims phone line in the benefits portal.
""".strip(),
    },
    {
        "id": "product_education_loan",
        "title": "Education Loan Product",
        "section": "Lending",
        "source": "product_catalog.pdf",
        "page": 11,
        "text": """
Education loans can be used to finance tuition fees and educational
expenses for an accredited college, university, or vocational program.
Eligible expenses are tuition, required books, laboratory fees, and
equipment the school lists as mandatory for the course.

The loan does not cover holidays, consumer electronics that the school
does not require, or living costs above the school's published allowance.
A student, or a parent who is the co-borrower, applies with an admission
letter and a fee schedule. Disbursements go to the school for tuition and
to the borrower for other approved expenses.

The interest rate is fixed for the study period and reviewed when the
repayment period starts. Repayment begins six months after the student
leaves the program. There is no penalty for paying the balance early.
""".strip(),
    },
    {
        "id": "product_home_loan",
        "title": "Home Loan Product",
        "section": "Lending",
        "source": "product_catalog.pdf",
        "page": 19,
        "text": """
Home loans are available for purchasing a residential property that the
borrower will live in. The property must be a completed house or apartment
in an area where the bank lends. The maximum loan is 80 percent of the
bank's valuation.

Investment properties, undeveloped land, and commercial buildings are not
financed under this product. The borrower needs proof of income and a
down payment from their own funds. The loan term can be 10, 15, 20, or
25 years. Applications are submitted at a branch or through the home
finance desk.
""".strip(),
    },
    {
        "id": "policy_parental",
        "title": "Parental Leave",
        "section": "Leave Policy",
        "source": "employee_handbook.pdf",
        "page": 18,
        "text": """
A primary caregiver receives 16 weeks of paid parental leave. A secondary
caregiver receives 6 weeks of paid parental leave. The leave can start up
to two weeks before the expected date of birth or adoption placement.

Parental leave is in addition to the annual leave balance. It must be
requested at least 30 days in advance when the date is known. During
parental leave the employee remains on the health insurance plan. Unused
parental leave cannot be converted to cash.
""".strip(),
    },
    {
        "id": "policy_security",
        "title": "Information Security",
        "section": "IT Policy",
        "source": "it_policy.pdf",
        "page": 4,
        "text": """
Every employee must use multi-factor authentication on email, the HR
portal, and the core banking systems. Passwords are not shared and are
not stored in documents or chat messages.

Personal USB drives are not allowed on company computers. Customer data
is not downloaded to a laptop except through the approved secure folder.
A suspected phishing message is reported to the security desk within one
hour. The message is forwarded as an attachment and then deleted. Employees
do not click links in unexpected payment or password-reset emails.
""".strip(),
    },
    {
        "id": "policy_conduct",
        "title": "Code of Conduct",
        "section": "Ethics",
        "source": "code_of_conduct.pdf",
        "page": 2,
        "text": """
Employees treat colleagues and customers with respect. Harassment, bullying,
and discrimination are not permitted and can lead to dismissal. A concern
can be raised with a manager or through the ethics mailbox.

Gifts from customers or suppliers above 50 dollars must be reported to
Compliance before they are accepted. Cash gifts are never accepted. An
employee does not approve a loan, a vendor, or a hire for a family member.
""".strip(),
    },
    {
        "id": "policy_expenses",
        "title": "Expense Approval Limits",
        "section": "Finance",
        "source": "finance_policies.pdf",
        "page": 15,
        "text": """
The reimbursement limit without extra approval is 2000 dollars per claim.
A manager may approve a claim at or below that amount when the receipts
match an allowed category. A claim above 2000 dollars also needs approval
from a director before Finance will pay it.

Split claims that hide a single purchase above the limit are not allowed.
Software subscriptions and training fees follow this same limit. Travel
claims follow the business travel policy and this approval limit together.
""".strip(),
    },
    {
        "id": "policy_onboarding",
        "title": "New Hire Onboarding",
        "section": "People",
        "source": "hr_sop.pdf",
        "page": 6,
        "text": """
A new employee receives a laptop and an identity badge on the first day.
IT enables the accounts before the start date. The hiring manager assigns
a buddy for the first 30 days. The buddy helps with tools, team routines,
and the first customer or internal meetings.

Mandatory training, including security awareness and conduct, is completed
within 14 days of the start date. Access to production systems stays
limited until that training is marked complete. Probation lasts three
months.
""".strip(),
    },
    {
        "id": "policy_performance",
        "title": "Performance Reviews",
        "section": "People",
        "source": "hr_sop.pdf",
        "page": 27,
        "text": """
Performance is reviewed twice a year, in March and in September. Goals are
set with the manager in January and can be adjusted in July if the role
changes. Each review covers outcomes, conduct, and development goals.

Ratings are recorded in the HR system. A rating is not shared with the
wider team. An employee who disagrees with a rating may ask Human Resources
for a second discussion within ten working days.
""".strip(),
    },
    {
        "id": "filing_q2",
        "title": "Quarterly Financial Filing",
        "section": "Investor Filing",
        "source": "q2_2026_filing.pdf",
        "page": 1,
        "text": """
Northwind Financial reported a 17.4% increase in net interest margin (NIM)
for the second quarter of 2026 compared with the same quarter a year
earlier. The filing attributes the change to a higher yield on loans and
a slower rise in deposit costs. The 17.4% NIM figure is the year-on-year
change in the margin, not the margin itself.
""".strip(),
    },
    {
        "id": "policy_tuition",
        "title": "Employee Tuition Assistance",
        "section": "Benefits",
        "source": "benefits_guide.pdf",
        "page": 9,
        "text": """
Full-time employees can receive up to 5000 dollars per calendar year
toward tuition for a course that is related to their current role or to
an agreed career plan. The course must be at an accredited institution.

The employee pays the school first and claims the cost after earning a
grade of C or better. The claim includes the invoice and the transcript.
Assistance is not available for hobbies, and it does not cover the
customer education loan product, which is a separate lending offer.
""".strip(),
    },
    {
        "id": "policy_training",
        "title": "Internal Training Deadlines",
        "section": "Learning",
        "source": "learning_handbook.pdf",
        "page": 5,
        "text": """
Staff enrolled in an internal course must submit assignments before the
published deadline. A deadline is shown in the learning portal when the
course opens. Work submitted late is not graded, except when the employee
has an approved leave that covers the due date.

Extensions are requested at least two days before the deadline. The course
owner decides. Completing assigned courses is part of the performance
review for roles that list them as mandatory.
""".strip(),
    },
    {
        "id": "policy_payroll",
        "title": "Payroll Schedule",
        "section": "Payroll",
        "source": "finance_policies.pdf",
        "page": 2,
        "text": """
Salaries are paid on the last business day of the month. When that day is
a public holiday, payment is made on the previous business day. Overtime
and approved reimbursements that Finance has cleared by the 15th are
included in the same payment. Items cleared after the 15th are paid the
following month.
""".strip(),
    },
    {
        "id": "policy_privacy",
        "title": "Customer Data Privacy",
        "section": "Compliance",
        "source": "privacy_policy.pdf",
        "page": 7,
        "text": """
Customer account numbers, balances, and contact details are used only for
the service the customer requested. Employees do not copy customer data
into personal email, personal chat, or an unapproved spreadsheet.

A customer may ask for a copy of the personal data the bank holds. The
privacy team responds within 30 days. A suspected data leak is reported
to the privacy office the same day it is discovered.
""".strip(),
    },
    {
        "id": "policy_branches",
        "title": "Branch Opening Hours",
        "section": "Branches",
        "source": "branch_operations.pdf",
        "page": 1,
        "text": """
Retail branches are open from 09:00 to 17:00, Monday to Friday, and from
09:00 to 13:00 on Saturday. Branches are closed on Sunday and on public
holidays. The cash desk stops accepting large cash deposits 30 minutes
before closing. Appointments for home loan applications can be booked
during opening hours.
""".strip(),
    },
    {
        "id": "policy_ethics",
        "title": "Speaking Up",
        "section": "Ethics",
        "source": "code_of_conduct.pdf",
        "page": 12,
        "text": """
Employees who see fraud, harassment, or a breach of policy can report it
to a manager, to Compliance, or to the ethics hotline. Reports to the
hotline can be anonymous. The company does not retaliate against someone
who raises a concern in good faith. Knowingly false reports are themselves
a conduct issue.
""".strip(),
    },
    {
        "id": "policy_retirement",
        "title": "Retirement Plan",
        "section": "Benefits",
        "source": "benefits_guide.pdf",
        "page": 16,
        "text": """
Full-time employees may join the retirement plan after probation. The
company matches employee contributions up to 6 percent of base salary.
The match is paid each month the employee contributes. Withdrawals follow
the plan rules and may be restricted before retirement age. The match rate
is reviewed annually by the benefits committee and is currently 6 percent.
""".strip(),
    },
]
