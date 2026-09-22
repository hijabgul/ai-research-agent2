    def _build_pdf(markdown_text: str) -> bytes:
        """Convert the markdown report into a PDF using fpdf2."""
        from fpdf import FPDF

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(15, 15, 15)       # tighter margins
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Fixed usable width -- avoids the "not enough horizontal space"
        # exception that happens when width=0 and the cursor is near the edge
        USABLE_WIDTH = pdf.epw - 10   # effective page width minus padding

        def _safe(text: str) -> str:
            """Strip non-Latin-1 chars + break very long words."""
            text = (
                text.replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2022", "-")
                .encode("latin-1", "replace")
                .decode("latin-1")
            )
            # Break any word longer than 60 chars so fpdf always
            # has a place to wrap
            out_words = []
            for w in text.split(" "):
                while len(w) > 60:
                    out_words.append(w[:60])
                    w = w[60:]
                out_words.append(w)
            return " ".join(out_words)

        for raw_line in markdown_text.splitlines():
            line = raw_line.rstrip()

            if line.startswith("|---") or (
                line.startswith("|") and set(line.replace("|", "").strip()) == {"-"}
            ):
                continue

            if line.startswith("# "):
                pdf.set_font("Helvetica", "B", 18)
                pdf.ln(2)
                pdf.multi_cell(USABLE_WIDTH, 10, _safe(line[2:]))
                pdf.ln(2)
                pdf.set_font("Helvetica", size=11)

            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 14)
                pdf.ln(3)
                pdf.multi_cell(USABLE_WIDTH, 8, _safe(line[3:]))
                pdf.ln(1)
                pdf.set_font("Helvetica", size=11)

            elif line.startswith("### "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(2)
                pdf.multi_cell(USABLE_WIDTH, 7, _safe(line[4:]))
                pdf.set_font("Helvetica", size=11)

            elif line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(USABLE_WIDTH, 6, _safe("   |   ".join(cells)))
                pdf.set_font("Helvetica", size=11)

            elif line.startswith("- ") or line.startswith("* "):
                pdf.multi_cell(USABLE_WIDTH, 6, _safe("  - " + line[2:]))

            elif line.strip() == "":
                pdf.ln(3)

            else:
                clean = _safe(line.replace("**", "").replace("*", ""))
                pdf.multi_cell(USABLE_WIDTH, 6, clean)

        return bytes(pdf.output())
