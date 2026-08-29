from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.lif_graphene_vasp.validate_vasprun import (
    VasprunValidationError,
    load_successful_vasprun,
)


def _vasprun_xml(*, nelm: int = 4, scf_steps: int = 2, close: bool = True) -> str:
    electronic_steps = "\n".join("<scstep />" for _ in range(scf_steps))
    closing_tag = "</modeling>" if close else ""
    return f"""<?xml version="1.0"?>
<modeling>
  <generator><i name="version" type="string">6.5.0</i></generator>
  <incar>
    <i name="NELM" type="int">{nelm}</i>
    <i name="NSW" type="int">0</i>
    <i name="IBRION" type="int">-1</i>
  </incar>
  <atominfo><atoms>2</atoms></atominfo>
  <calculation>
    {electronic_steps}
    <varray name="forces">
      <v>0.1 0.0 -0.1</v>
      <v>-0.1 0.0 0.1</v>
    </varray>
    <energy><i name="e_0_energy">-3.25</i></energy>
  </calculation>
{closing_tag}
"""


class VasprunValidatorTests(unittest.TestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / "vasprun.xml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_static_summary_uses_no_pymatgen_types(self) -> None:
        with TemporaryDirectory() as directory:
            run = load_successful_vasprun(
                self._write(directory, _vasprun_xml()),
                expected_atoms=2,
                require_static=True,
            )
        self.assertEqual(run.vasp_version, "6.5.0")
        self.assertEqual(run.final_energy, -3.25)
        self.assertEqual(len(run.ionic_steps[-1]["electronic_steps"]), 2)
        self.assertEqual(len(run.ionic_steps[-1]["forces"]), 2)

    def test_exact_nelm_steps_are_rejected_as_unconverged(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, _vasprun_xml(nelm=2, scf_steps=2))
            with self.assertRaises(VasprunValidationError) as caught:
                load_successful_vasprun(path, require_static=True)
        self.assertEqual(caught.exception.status, "unconverged_electronic")

    def test_missing_closing_tag_is_reported_before_xml_parsing(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, _vasprun_xml(close=False))
            with self.assertRaises(VasprunValidationError) as caught:
                load_successful_vasprun(path)
        self.assertEqual(caught.exception.status, "incomplete_xml")

    def test_atom_count_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, _vasprun_xml())
            with self.assertRaises(VasprunValidationError) as caught:
                load_successful_vasprun(path, expected_atoms=3)
        self.assertEqual(caught.exception.status, "wrong_atom_count")


if __name__ == "__main__":
    unittest.main()
