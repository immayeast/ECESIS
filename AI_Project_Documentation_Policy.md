# AI Project Documentation Policy

## Core Principle

> The AI proposes. The human decides. The repository records.

## Purpose

This policy standardizes how AI assistants participate in software engineering, data science, and research projects. The goal is to ensure transparency, reproducibility, and clear human ownership of all decisions.

## Required Repository Structure

```text
project/
├── README.md
├── progress.md
├── decisions.md
├── ai_journal/
├── prompts/
├── adr/
├── src/
├── tests/
└── outputs/
```

## File Responsibilities

### progress.md
Chronological record of milestones and completed work.

### decisions.md
Human decisions, alternatives considered, and rationale.

### ai_journal/YYYY-MM-DD.md
Daily record of AI usage, accepted/rejected suggestions, and verification.

### prompts/
Store prompts and summaries of high-impact AI outputs.

### adr/
Architecture Decision Records for major design choices.

## AI Contribution Levels

| Level | Meaning |
|------:|---------|
| 0 | No AI used |
| 1 | Brainstorming only |
| 2 | AI proposed approach |
| 3 | AI generated substantial draft code or text |
| 4 | AI produced most of the artifact |

## Daily AI Journal Template

```markdown
# AI Journal — YYYY-MM-DD

## Goal
What problem was being solved?

## Tools Used
Which AI tools were used?

## Prompts
Exact prompts or summaries.

## Suggestions
Key ideas proposed by the AI.

## Accepted
What was adopted.

## Rejected
What was discarded and why.

## Human Modifications
How the final solution differed.

## Verification
How correctness was tested.

## Files Modified
- src/example.py

## AI Contribution Level
2
```

## Architecture Decision Record Template

```markdown
# ADR-XXX: Title

Status: Accepted
Date: YYYY-MM-DD

## Context
What problem needed a decision?

## Alternatives
- Option A
- Option B

## Decision
Chosen approach.

## Consequences
Benefits and trade-offs.
```

## Assistant Responsibilities

Any AI assistant working in the repository should:

1. Follow repository conventions.
2. Update `progress.md` after meaningful milestones.
3. Record major reasoning in `decisions.md`.
4. Create an `ai_journal` entry when AI materially influences work.
5. Save important prompts in `prompts/`.
6. Create ADRs for significant architecture decisions.
7. Clearly identify assumptions and uncertainties.
8. Encourage verification and testing.
9. Never represent unverified output as fact.

## Git Commit Convention

```bash
git commit -m "Implement risk score (AI-assisted; see ai_journal/2026-05-17.md)"
```

## Recommended Workflow

1. Define the task.
2. Consult AI.
3. Evaluate suggestions.
4. Implement and test.
5. Record progress.
6. Record decisions.
7. Log AI usage.
8. Commit changes.

## Guiding Principles

- Human ownership of all decisions.
- Reproducibility over convenience.
- Transparency over opacity.
- Verification over assumption.
- Documentation as a first-class artifact.
