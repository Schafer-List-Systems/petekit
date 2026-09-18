"""PdfTranscriber - high-quality PDF transcription via tesseract + LLM vision."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time as time_mod
from dataclasses import dataclass
from pathlib import Path

from peteos.oap.agentic_object import AgenticObject, tool, Error

from .buffer_manager import BufferManager


class PdfTranscriber(BufferManager, AgenticObject):
    """You are a PDF page transcriber.

    You load PDF files into memory buffers and transcribe their contents.
    Use transcribe_pdf to process a PDF and store its transcription in a buffer.

    Workflow:
      1. transcribe_pdf(pdf_path) — transcribe a PDF and store text in a buffer
      2. read_buffer(buffer_name, start=N, end=M) — read a range of transcribed pages
      3. grep_buffer(buffer_name, "regex") — search across transcribed text

    Use (?i) at the start of a grep pattern for case-insensitive matching.
    """

    @tool(description="Transcribe a PDF and store its text in a buffer. Each page becomes one line. Returns a hint with the buffer name and page count.")
    async def transcribe_pdf(self, pdf_path: str, cross_reference: bool = False) -> str:
        """Transcribe a PDF, store results in a buffer, return a status hint."""
        abs_path = Path(pdf_path).resolve()
        if not abs_path.exists():
            return f"Error: file '{pdf_path}' not found."

        tmpdir = tempfile.mkdtemp(prefix="pdf_transcribe_")
        tmpdir_path = Path(tmpdir)

        try:
            self._split_pdf(str(abs_path), tmpdir_path)
            results: list[tuple[str, str]] = []
            pages = sorted(tmpdir_path.glob("page-*.pdf"))
            if not pages:
                return f"Error: no pages found in PDF '{pdf_path}'."

            for page_pdf in pages:
                page_name = page_pdf.stem
                png_path = tmpdir_path / f"{page_name}.png"
                self._render_page(page_pdf, png_path)
                tesseract_text = self._tesseract_transcribe(str(png_path))
                if cross_reference:
                    agent_result = await self._vision_agent_transcribe(str(png_path))
                    image_description = agent_result.image_description
                    unified_text = await self._cross_reference(
                        tesseract_text,
                        agent_result.text,
                    )
                else:
                    unified_text = tesseract_text
                    image_description = ""
                page_pdf.unlink(missing_ok=True)
                png_path.unlink(missing_ok=True)
                results.append((unified_text, image_description))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        buffer_name = f"pdf:{abs_path}"
        text = "\n".join([
            f"=== Page {i+1} ===\n{text}\n### Image Description: {desc}"
            for i, (text, desc) in enumerate(results)
        ])
        self.create_buffer(buffer_name, text=text)
        return (
            f"Transcribed {len(results)} page(s) from '{pdf_path}' into buffer '{buffer_name}'. "
            f"Use read_buffer to access the content."
        )

    @staticmethod
    def _split_pdf(pdf_path: str, tmpdir: Path) -> None:
        """Split PDF into single-page PDFs using mutool."""
        info = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        num_pages = None
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                num_pages = int(line.split(":")[1].strip())
                break
        if num_pages is None:
            raise ValueError("Could not determine page count from pdfinfo")
        for i in range(1, num_pages + 1):
            name = f"page-{i:03d}"
            subprocess.run(
                ["mutool", "merge", "-o", str(tmpdir / f"{name}.pdf"),
                 "-O", "garbage=compact", pdf_path, str(i)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

    @staticmethod
    def _render_page(page_pdf: Path, png_path: Path) -> None:
        """Convert a single-page PDF to PNG using mutool."""
        subprocess.run(
            ["mutool", "draw", "-F", "png", "-w", "1600", "-r", "150",
             "-o", str(png_path), str(page_pdf), "1"],
            check=True,
            timeout=30,
        )

    @staticmethod
    def _tesseract_transcribe(png_path: str) -> str:
        """Transcribe a PNG image using tesseract OCR."""
        try:
            result = subprocess.run(
                ["tesseract", png_path, "-", "--psm", "6"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            return result.stdout
        except FileNotFoundError:
            return ""
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    async def _vision_agent_transcribe(self, png_path: str) -> "TranscriptionResult":
        """Run the LLM vision agent for text transcription + image description."""
        prompt = (
            "Transcribe the text content of this PDF page exactly as it appears. "
            "Also describe ALL images, tables, charts, or diagrams visible on the page.\n\n"
        )
        output = await self.invoke_agent(
            prompt=prompt,
            image=png_path,
            output_schema=TranscriptionResult,
            persistent_thread_id=None,
            timeout=None,
        )
        if isinstance(output, Error):
            raise RuntimeError(output.message)
        return output

    async def _cross_reference(self, tesseract_text: str, vision_text: str) -> str:
        """Cross-reference tesseract and LLM vision transcriptions."""
        prompt = (
            f"Tesseract OCR result:\n{tesseract_text}\n\n"
            f"LLM vision transcription:\n{vision_text}\n\n"
            "Cross-reference both sources and produce a single unified "
            "transcription. Preserve all text that appears in either source. "
            "If the two disagree, prefer the more complete and accurate version. "
            "Return only the unified text, nothing else. "
        )
        result = await self.invoke_agent(
            prompt=prompt,
            output_schema=UnifiedText,
            persistent_thread_id=None,
            timeout=None,
        )
        if isinstance(result, Error):
            raise RuntimeError(result.message)
        return result.text


@dataclass
class TranscriptionResult:
    """Structured output from the vision agent transcription step."""
    text: str
    image_description: str


@dataclass
class UnifiedText:
    """Structured output from the cross-reference step."""
    text: str
