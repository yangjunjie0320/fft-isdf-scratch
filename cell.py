import os, sys
from mp_api.client import MPRester as M

API_KEY = os.getenv("MP_API_KEY", None)
assert API_KEY is not None, "MP_API_KEY is not set"

TMP_DIR = os.getenv("TMP_DIR", "/tmp")
assert os.path.exists(TMP_DIR), f"TMP_DIR {TMP_DIR} does not exist"

def download_poscar(mid: str, name=None, path=None) -> str:
    if path is None:
        path = TMP_DIR

    if name is None:
        name = f"{mid}"

    # Initialize the Materials Project REST API client
    with M(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mid)
        poscar_path = os.path.join(path, f"{name}.vasp")
        structure.to(filename=str(poscar_path), fmt="poscar")
        print(f"Successfully downloaded POSCAR for {mid} to {poscar_path}")
        return str(poscar_path)

# Example usage:
if __name__ == "__main__":
    poscar_file = download_poscar("mp-19009", name="nio-2")

    import ase
    atoms = ase.io.read(poscar_file)
    print(atoms)

    from ase.build import bulk
    atoms = bulk("NiO", "rocksalt", a=4.18)
    atoms.write(os.path.join(TMP_DIR, "nio-1.vasp"))
