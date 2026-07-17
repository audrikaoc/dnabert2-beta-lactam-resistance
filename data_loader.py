#!/usr/bin/env python3
"""Parse a single CARD FASTA file and build a DataFrame of sequences and labels.

This script reads `nucleotide_fasta_protein_homolog_model.fasta` (or any provided
FASTA file), filters Escherichia coli sequences into resistant vs control groups
based on header keywords, strips non-ATCG characters from sequences, and
outputs a Pandas DataFrame with `sequence` and `label` columns.

Extensive inline comments are included for clarity, as requested.
"""

# Standard library imports
import argparse  # command-line argument parsing
import os  # file existence checks and directory handling
import re  # regular expressions for text matching and sequence cleaning
import shutil  # locate BLAST executables on the system
import subprocess  # run BLAST commands from Python
import tempfile  # create temporary files/directories for BLAST input/output
from pathlib import Path  # filesystem path handling
from typing import List, Optional, Set, Tuple  # type hints for function signatures

# Third-party imports
import pandas as pd  # DataFrame construction and manipulation
from Bio import SeqIO  # Biopython sequence IO for FASTA parsing


def is_resistant(header: str, resist_keywords: List[str]) -> bool:
	"""Return True if header indicates a resistant Escherichia coli sequence.

	The function checks that the organism `Escherichia coli` appears in the
	header and that at least one of the resistance keywords (e.g., 'TEM',
	'CTX-M', 'ampC') is present. Matching is case-insensitive.
	"""
	# Lowercase the header to perform case-insensitive checks
	header_lower = header.lower()
	# Quick check: must mention Escherichia coli (case-insensitive)
	if "escherichia coli" not in header_lower:
		return False
	# If any resistance keyword appears in the header, mark as resistant
	for kw in resist_keywords:
		if kw.lower() in header_lower:
			return True
	return False


def is_control(header: str, control_keywords: List[str], exclude_keywords: List[str]) -> bool:
	"""Return True if header indicates a control (non-beta-lactamase) sequence.

	This function requires `Escherichia coli` to appear in the header and
	looks for known housekeeping or control gene names (e.g., 'recA', 'rpoB').
	It also ensures control headers do not accidentally contain resistance
	keywords listed in `exclude_keywords`.
	"""
	# Lowercase header for case-insensitive checks
	header_lower = header.lower()
	# Require organism to be E. coli in the header
	if "escherichia coli" not in header_lower:
		return False
	# If header contains any exclusion (resistance) keyword, do not treat as control
	for ek in exclude_keywords:
		if ek.lower() in header_lower:
			return False
	# If any control keyword is present, it's a control
	for ck in control_keywords:
		if ck.lower() in header_lower:
			return True
	return False


def clean_sequence(seq: str) -> str:
	"""Return the DNA sequence with only A, C, G, T characters (uppercase).

	Any character not in the set {A,C,G,T,a,c,g,t} will be removed. This
	strips ambiguous bases like 'N' and any annotation characters that may be
	present in the sequence string.
	"""
	# Use a regular expression to remove any character that is not A/C/G/T (case-insensitive)
	cleaned = re.sub(r"[^ACGTacgt]", "", seq)
	# Convert to uppercase to standardize sequences
	return cleaned.upper()


def verify_sequences_with_blast(
	query_fasta_path: str,
	reference_fasta_path: str,
	evalue: float = 1e-6,
	identity_threshold: float = 80.0,
	min_alignment_length: int = 100,
	output_dir: Optional[str] = None,
) -> Set[str]:
	"""Use BLASTN to keep query sequences that match a reference E. coli FASTA.

	This function writes a temporary BLAST database from the supplied reference
	FASTA and runs BLASTN against a query FASTA. It returns the identifiers of
	queries that meet the identity/length/e-value thresholds. If BLAST+ tools
	are missing, it raises a clear runtiaume error so the caller can decide how
	to proceed.
	"""
	# Check that the query and reference files exist before running BLAST
	if not os.path.exists(query_fasta_path):
		raise FileNotFoundError(f"Query FASTA not found: {query_fasta_path}")
	if not os.path.exists(reference_fasta_path):
		raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta_path}")

	# Find BLAST executables on the current system
	blastn = shutil.which("blastn")
	makeblastdb = shutil.which("makeblastdb")
	if not blastn or not makeblastdb:
		raise RuntimeError(
			"BLAST+ executables were not found. Install BLAST+ and rerun this step."
		)

	# Create a temporary directory for database and result files.
	# Use the system temporary directory by default to avoid issues with
	# spaces in the current working directory path.
	temp_dir = Path(output_dir or tempfile.mkdtemp(prefix="blast_verify_"))
	temp_dir.mkdir(parents=True, exist_ok=True)
	db_path = temp_dir / "reference"
	result_path = temp_dir / "blast_results.tsv"

	# Build a temporary nucleotide BLAST database from the reference FASTA
	makeblastdb_cmd = [
		makeblastdb,
		"-in",
		reference_fasta_path,
		"-dbtype",
		"nucl",
		"-out",
		str(db_path),
	]
	subprocess.run(makeblastdb_cmd, check=True, capture_output=True, text=True)

	# Run BLASTN against the query FASTA and request a tab-delimited summary
	blast_cmd = [
		blastn,
		"-query",
		query_fasta_path,
		"-db",
		str(db_path),
		"-outfmt",
		"6 qseqid sseqid pident length evalue bitscore",
		"-max_target_seqs",
		"1",
		"-evalue",
		str(evalue),
	]
	blast_result = subprocess.run(blast_cmd, check=True, capture_output=True, text=True)
	result_path.write_text(blast_result.stdout)

	# Parse BLAST output and keep only hits that pass the thresholds
	verified_ids: Set[str] = set()
	for line in blast_result.stdout.splitlines():
		if not line.strip():
			continue
		parts = line.split("\t")
		if len(parts) < 6:
			continue
		query_id = parts[0]
		percent_identity = float(parts[2])
		alignment_length = int(float(parts[3]))
		blast_evalue = float(parts[4])
		if (
			percent_identity >= identity_threshold
			and alignment_length >= min_alignment_length
			and blast_evalue <= evalue
		):
			verified_ids.add(query_id)

	return verified_ids


def parse_fasta_to_dataframe(
	fasta_path: str,
	resist_keywords: List[str],
	control_keywords: List[str],
	exclude_keywords: List[str],
	blast_reference_fasta: Optional[str] = None,
	blast_identity_threshold: float = 80.0,
	blast_min_alignment_length: int = 100,
	blast_evalue: float = 1e-6,
	blast_output_dir: Optional[str] = None,
	include_unverified_controls: bool = False,
) -> pd.DataFrame:
	"""Parse the FASTA file and return a DataFrame with `sequence` and `label`.

	The function iterates over each FASTA record, checks the header for
	resistant vs control signals, cleans the sequence, and appends labeled
	entries. Sequences that are ambiguous from the header alone can be verified
	with BLAST against a reference E. coli FASTA and kept as controls (0) if they
	match the reference well enough.
	"""
	# Prepare lists to accumulate sequences, labels, and verification flags
	sequences: List[str] = []
	labels: List[int] = []
	verified_flags: List[bool] = []

	# Store ambiguous E. coli records separately so they can be checked with BLAST
	pending_records: List[Tuple[str, str, str]] = []

	# Use Biopython SeqIO to iterate through FASTA records efficiently
	for record in SeqIO.parse(fasta_path, "fasta"):
		# Extract the header/description text from the record
		header = record.description
		# Quick organism filter: only E. coli records are considered
		if "escherichia coli" not in header.lower():
			continue

		# Convert the Seq object to a plain string and clean it
		raw_seq = str(record.seq)
		cleaned = clean_sequence(raw_seq)
		# Skip records that become empty after cleaning
		if not cleaned:
			continue

		# Assign label logic based on the header first
		header_lower = header.lower()
		if any(kw.lower() in header_lower for kw in resist_keywords):
			sequences.append(cleaned)
			labels.append(1)
			verified_flags.append(True)
		elif any(ck.lower() in header_lower for ck in control_keywords):
			sequences.append(cleaned)
			labels.append(0)
			verified_flags.append(True)
		else:
			# Ambiguous E. coli entry: neither resistance nor control keywords found.
			# Store it for optional BLAST verification instead of discarding it.
			pending_records.append((record.id, header, cleaned))

	# If BLAST verification was requested, check the pending sequences once.
	if blast_reference_fasta and pending_records:
		# Create a temporary FASTA file containing only the ambiguous E. coli sequences
		with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as handle:
			for record_id, header, cleaned_seq in pending_records:
				handle.write(f">{record_id} {header}\n{cleaned_seq}\n")
			query_fasta_path = handle.name

		try:
			verified_ids = verify_sequences_with_blast(
				query_fasta_path=query_fasta_path,
				reference_fasta_path=blast_reference_fasta,
				evalue=blast_evalue,
				identity_threshold=blast_identity_threshold,
				min_alignment_length=blast_min_alignment_length,
				output_dir=blast_output_dir,
			)
		except RuntimeError as exc:
			print(f"BLAST verification skipped: {exc}")
			verified_ids = set()
		finally:
			# Remove the temporary query FASTA after the BLAST run is complete
			if os.path.exists(query_fasta_path):
				os.remove(query_fasta_path)

		# Keep the BLAST-verified sequences as label 0 controls.
		for record_id, header, cleaned_seq in pending_records:
			if record_id in verified_ids:
				sequences.append(cleaned_seq)
				labels.append(0)
				verified_flags.append(True)
			elif include_unverified_controls:
				sequences.append(cleaned_seq)
				labels.append(0)
				verified_flags.append(False)

	# If BLAST verification was not requested, optionally keep all pending E. coli sequences
	# as unverified controls when the user wants to retain non-resistant candidates.
	elif include_unverified_controls:
		for record_id, header, cleaned_seq in pending_records:
			sequences.append(cleaned_seq)
			labels.append(0)
			verified_flags.append(False)

	# Build a Pandas DataFrame from the collected sequences, labels, and verification flags
	df = pd.DataFrame({"sequence": sequences, "label": labels, "verified": verified_flags})
	return df


def main() -> None:
	"""Command-line entrypoint for the script.

	The CLI accepts a FASTA file path and optional output CSV path. By default
	the script uses a set of common resistance and housekeeping keywords,
	which can be overridden by providing comma-separated lists.
	"""
	# Set up command-line arguments for flexibility when running the script
	parser = argparse.ArgumentParser(
		description=(
			"Parse a CARD FASTA and produce a DataFrame of E. coli sequences "
			"labeled for beta-lactam resistance (1) vs controls (0)."
		)
	)

	# Positional argument: path to the FASTA file to parse
	parser.add_argument("fasta", help="Path to input FASTA file")
	# Optional output path to save the DataFrame as CSV
	parser.add_argument(
		"--output",
		"-o",
		help="Optional path to save the resulting CSV (sequence,label)",
		default=None,
	)
	# Allow the user to specify additional control keywords via CSV
	parser.add_argument(
		"--extra-controls",
		help=(
			"Comma-separated list of additional control gene keywords to treat "
			"as non-resistance controls (e.g., 'abc,def')."
		),
		default="",
	)
	# Optional BLAST reference FASTA for verifying ambiguous E. coli sequences
	parser.add_argument(
		"--blast-reference",
		help="Path to a reference FASTA for BLAST-based E. coli verification",
		default=None,
	)
	parser.add_argument(
		"--blast-identity",
		type=float,
		help="Minimum percent identity for a BLAST hit to retain a sequence",
		default=80.0,
	)
	parser.add_argument(
		"--blast-min-length",
		type=int,
		help="Minimum alignment length in bases for a BLAST hit to retain a sequence",
		default=100,
	)
	parser.add_argument(
		"--blast-evalue",
		type=float,
		help="Maximum E-value for a BLAST hit to retain a sequence",
		default=1e-6,
	)
	parser.add_argument(
		"--blast-output-dir",
		help="Optional directory to store BLAST output files",
		default=None,
	)
	parser.add_argument(
		"--include-unverified-controls",
		action="store_true",
		help="Include non-resistant E. coli sequences even if BLAST verification fails",
	)

	# Parse CLI args into variables
	args = parser.parse_args()

	# Default resistance keywords (common beta-lactamase families)
	default_resist = ["TEM", "CTX-M", "ampC"]

	# Default control/houskeeping gene keywords (common examples)
	default_controls = ["recA", "rpoB", "gyrA", "atpD", "fusA", "rpsL"]

	# If the user provided extra controls, extend the default list
	extra_controls = [x.strip() for x in args.extra_controls.split(",") if x.strip()]
	control_keywords = default_controls + extra_controls

	# For exclusion when identifying controls, use the same resistance keywords
	exclude_keywords = default_resist

	# Parse the FASTA and build the DataFrame
	df = parse_fasta_to_dataframe(
		fasta_path=args.fasta,
		resist_keywords=default_resist,
		control_keywords=control_keywords,
		exclude_keywords=exclude_keywords,
		blast_reference_fasta=args.blast_reference,
		blast_identity_threshold=args.blast_identity,
		blast_min_alignment_length=args.blast_min_length,
		blast_evalue=args.blast_evalue,
		blast_output_dir=args.blast_output_dir,
		include_unverified_controls=args.include_unverified_controls,
	)

	# Print a short summary of counts for quick feedback
	counts = df["label"].value_counts().to_dict()
	print("Label counts:", counts)

	# Optionally save to CSV if an output path was provided
	if args.output:
		df.to_csv(args.output, index=False)
		print(f"Saved labeled sequences to {args.output}")
	else:
		# If no output path, show the head of the DataFrame for inspection
		print(df.head())


if __name__ == "__main__":
	# Run main() when the script is executed directly
	main()

