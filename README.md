# LLM-Evaluations-for-Bharat-Tax-Laws
A specialized tool for evaluating and benchmarking AI models' performance on Indian tax-related queries using multiple evaluation metrics.

## Overview

This tool compares AI model responses against ground truth answers for tax-related queries, evaluating them across six key metrics:
- Issue Identification
- Rule Identification
- Application of Law
- Conclusion Correctness
- Interpretation/Reasoning
- Argumentation/Justification

## Prerequisites

- Python 3.x
- Google's Generative AI SDK
- Required Python packages (install via pip):
  - pandas
  - tqdm
  - google-generativeai
  - openpyxl or xlrd (for Excel file handling)

## Project Structure

```
├── tax_ai_benchmarking_new.py    # Main benchmarking script
├── run_benchmark.bat             # Windows batch script for Gemini model
```

## Usage

1. Prepare your input Excel files:
   - Ground truth file with "Query" and "Response" columns
   - Candidate response files with matching "Query" and "Response" columns

2. Run the benchmarking tool:
```bat
python tax_ai_benchmarking_new.py ^
  --gt "<path_to_ground_truth>.xlsx" ^
  --candidate "Name1=<path_to_candidate1>.xlsx" "Name2=<path_to_candidate2>.xlsx" ^
  --out "results.xlsx" ^
  --max-workers 10 ^
  --llm "model-name"
```

## Command Line Arguments

- `--gt`: Path to ground truth Excel file
- `--candidate`: One or more NAME=FILE pairs for candidate responses
- `--out`: Output Excel workbook path (default: evaluation_results_YYYYMMDD_HHMMSS.xlsx)
- `--llm`: LLM model name
- `--max-workers`: Maximum concurrent LLM requests (default: 5)

## Output Format

The tool generates an Excel workbook with multiple sheets:
- Detailed_{CandidateName}: Per-query scores for each candidate
- Summary: Overall performance metrics
- Leaderboard: Ranked comparison of all candidates

## Evaluation Metrics

Each response is evaluated on a 1-5 scale across six dimensions:

1. **Issue Identification** (IssueId)
   - Evaluates precision in identifying legal/factual controversies
   - Requires clear, question-shaped issue framing

2. **Rule Identification** (RuleId)
   - Assesses coverage of relevant statutory provisions
   - Checks citation accuracy and completeness

3. **Application of Law** (ApplyLaw)
   - Measures how rules are applied to specific facts
   - Evaluates reasoning depth and logical structure

4. **Conclusion Correctness** (Conclusion)
   - Validates logical flow from analysis to conclusion
   - Checks for clarity and legal supportability

5. **Interpretation** (Interpretation)
   - Assesses ability to parse amendments and legal documents
   - Evaluates linguistic and legal nuance comprehension

6. **Justification** (Justification)
   - Measures construction of legally persuasive arguments
   - Evaluates logical soundness and fact-sensitivity

## License

This project is licensed under the MIT License — see the accompanying `LICENSE` file for details.

