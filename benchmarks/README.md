# PDF to Word Benchmark

Suggested structure:

```text
benchmarks/
  README.md
  case_01/
    source.pdf
    expected.txt
```

Run:

```powershell
python tools/benchmark_pdf_to_word.py --pdf benchmarks/case_01/source.pdf --expected benchmarks/case_01/expected.txt --use-ocr
```

Interpretation:

- `DOCX text similarity` is the main score to optimize.
- Start with one representative Vietnamese PDF.
- Tune the pipeline until the case reaches the target before expanding to more files.
