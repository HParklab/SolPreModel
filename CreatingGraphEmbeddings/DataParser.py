import numpy as np
from typing import Dict, Tuple

class DataParser:
    def __init__(self, index_base='auto', return_base=0):
        """
        index_base: 'auto' | 0 | 1 -> 파일에서 atom index가 0-based인지 1-based인지
        return_base: 최종 반환 시 index base (0 or 1)
        """
        self.index_base = index_base
        self.return_base = return_base

    # ------------------------------------------------------------
    # XYZ Coordinate Parser
    # ------------------------------------------------------------
    def load_xyz_coords(self, xyz_path):
        coords = []
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 3:
            raise ValueError(f"XYZ file {xyz_path} is too short")
        
        first = lines[0].lstrip('\ufeff').strip()
        try:
            natoms = int(first.split()[0])
        except Exception as e:
            raise ValueError(f"First line of XYZ must be atom count, got '{first}") from e
        
        if len(lines) < 2 + natoms:
            raise ValueError(f"XYZ file {xyz_path} incomplete: natoms={natoms} but only {len(lines)-2} atom lines.")
        
        coords = []
        for i in range(natoms):
            raw = lines[2+i].strip()
            parts = raw.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed atom line at {i+3}: '{raw}'")
            try:
                x, y, z = map(float, parts[-3:])
            except Exception:
                # 혹시나를 대비한 폴백: 심볼 뒤 3개
                x, y, z = map(float, parts[1:4])
            coords.append([x, y, z])

        if len(coords) != natoms:
            raise ValueError(f"Expected {natoms} atoms but parsed {len(coords)} lines.")

        return np.array(coords, dtype=np.float32)

    # ------------------------------------------------------------
    # Cosmo Data Parser
    # ------------------------------------------------------------
    def load_cosmo(self, cosmo_path):
        """
        Parse .cosmo file's $segment_information for grid segments.
        Returns:
            grid_coodrs: np.ndarray[N_g, 3]
            origin_atoms: List[int] (0-based atom indices)
            sigma_vals: np.ndarray[N_g, 1] (surface charge density)
        """
        grid_coords = []
        origin_atoms = []
        sigma_vals = []
        in_seg = False
        bohr_to_ang = 0.529177

        with open(cosmo_path, 'r') as f:
            for line in f:
                if line.startswith("$segment_information"):
                    in_seg = True
                    continue
                if in_seg and (line.strip() == '' or line.startswith('$')):
                    break
                if in_seg:
                    tok = line.split()
                    atom_idx = int(tok[1]) -1 # 1-based -> 0-based
                    xyz = (np.asarray(tok[2:5], dtype=np.float64)*bohr_to_ang).tolist() #coordinates
                    sigma = float(tok[5])
                    origin_atoms.append(atom_idx)
                    grid_coords.append(xyz)
                    sigma_vals.append([sigma])
        return(
            np.array(grid_coords, dtype=np.float32),
            origin_atoms,
            np.array(sigma_vals, dtype=np.float32)
        )

    # ------------------------------------------------------------
    # Wiberg Bond Order(WBO) Parser
    # ------------------------------------------------------------
    def load_wbo(self, wbo_path: str, make_symmetric: bool = True, dedup: str = 'max') -> Dict[Tuple[int,int], float]:
        """
        Parse a Wiberg Bond Order text file where each non-comment line is: i j value.
            - make_symmetric: also write (j, i)
            - dedup: how to merge duplicates: 'max' | 'min' | 'last' | 'avg'
        Returns dict[(i,j), float] 
        """
        wbo_dict: Dict[Tuple[int, int], float] = {}
        counts: Dict[Tuple[int, int], int] = {}

        # probe first valid line to detect auto_detect index base
        first_pair = None
        with open(wbo_path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#') or s.lower().startswith('atom'):
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    i0 = int(parts[0]); j0 = int(parts[1])
                    first_pair = (i0, j0)
                    break
                except ValueError:
                    continue
    
        def to_zero(i: int) -> int:
            if self.index_base == '0' or self.index_base == 0:
                return i
            if self.index_base == '1' or self.index_base == 1:
                return i-1
            if first_pair is not None and 0 in first_pair:
                return i
            return i-1
        
        def put(i: int, j: int, v:float):
            key = (i, j)
            if key in wbo_dict:
                if dedup == 'max':
                    wbo_dict[key] = max(wbo_dict[key], v)
                elif dedup == 'min':
                    wbo_dict[key] = min(wbo_dict[key], v)
                elif dedup == 'last':
                    wbo_dict[key] = v
                elif dedup == 'avg':
                    n = counts.get(key, 1)
                    wbo_dict[key] = (wbo_dict[key]*n +v) / (n+1)
                    counts[key] = n+1
            else:
                wbo_dict[key] = v
                if dedup == 'avg':
                    counts[key] = 1

        with open(wbo_path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#') or s.lower().startswith('atom'):
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    i = to_zero(int(parts[0])); j = to_zero(int(parts[1]))
                    if i == j:
                        continue
                    v = float(parts[2])
                except ValueError:
                    continue
                put(i, j, v)
                if make_symmetric:
                    put(j, i, v)
        return wbo_dict

    # ------------------------------------------------------------
    # xTB Log Parser: per-atom charges (q)
    # ------------------------------------------------------------

    def load_xtb_charges(self, log_path: str):
        """
        Parse xTB xtdout log to extract per-atom partial charges from the block
        head by a line containing both 'convCN' and 'q'. Robust to spacing and scientific notation.
        Returns (charges, Zmap, symmap) dicts.
        """
        import re

        num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        row_re = re.compile(
            rf"^\s*(\d+)\s+(\d+)\s+([A-Za-z]+)\s+({num})\s+({num})\s+({num})\s+({num})\s*$"
        )

        with open(log_path, 'r', errors = 'ignore') as f:
            lines = f.read().splitlines()

        # locate header
        start = None
        for i, line in enumerate(lines):
            if ('convCN' in line) and re.search(r'\bconvCN\b', line) and re.search(r'\bq\b', line):
                start = i + 1
                break
        if start is None:
            raise ValueError(f"charge table not found in {log_path}")
        
        charges: Dict[int, float] = {}
        Zmap: Dict[int, int] = {}
        symmap: Dict[int,str] = {}
        first_idx = None

        i = start
        while i < len(lines):
            s = lines[i].rstrip()
            m = row_re.match(s)
            if not m:
                if s.strip() == '' or s.lstrip().startswith('Mol.'):
                    break
                break
            idx_str, Z_str, sym, convCN, q, C6AA, alpha0 = m.groups()
            idx = int(idx_str)
            if first_idx is None:
                first_idx = idx
            charges[idx] = float(q)
            Zmap[idx] = int(Z_str)
            symmap[idx] = sym
            i += 1

        # index base normalize
        if self.index_base == 'auto':
            src_base = 0 if first_idx == 0 else 1
        else:
            src_base = int(self.index_base)
        if self.return_base not in (0,1):
            raise ValueError("return_base must be 0 or 1")
        if src_base != self.return_base:
            shift = -1 if (src_base == 1 and self.return_base == 0) else 1
            charges = {k + shift: v for k, v in charges.items()}
            Zmap = {k + shift: v  for k, v in Zmap.items()}
            symmap = {k + shift: v for k, v in symmap.items()}

        return charges, Zmap, symmap

    # -----------------------------------------------------------------------------
    # xTB Log Parser: free energy of solvation(Gsolv), molecular quadrupole(full)
    # -----------------------------------------------------------------------------
    def load_global_features(self, log_path: str):
        """
        Parse xTB stdout log to extract free energy of solvation(Gsolv) and molecular 
        quadrupole moment(full) from the block head by a line containing "SUMMARY" and 
        "molecular quadrupole (traceless):".
        Returns gsolv(scalar) and quad_full(vec, dim=6).
        """
        import re
        from pathlib import Path

        num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)"

        re_quad_full = re.compile(
            rf"molecular quadrupole.*?\n.*\n\s*full:\s*",
            rf"({num})\s+({num})\s+9({num})\s+({num})\s+({num})\s+({num})",
            re.S
        )
        re_gsolv = re.compile(rf"->\s*Gsolv\s+({num})\s+Eh")

        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

        m_q = re_quad_full.search(text)
        if not m_q:
            raise ValueError("quadrupole full을 찾지 못함")
        quad_vec = np.array([float(g) for g in m_q.groups()], dtype=np.float64)

        m_g = re_gsolv.search(text)
        if not m_g:
            raise ValueError("Gsolv를 찾지 못함")
        gsolv = float(m_g.group(1))

        return quad_vec, gsolv
