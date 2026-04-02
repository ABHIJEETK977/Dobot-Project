## Changes in the file struce for clean use 

## **Folder Structure**

```
Conveyor belt pickup/
├── main_out.py
└── dobot_lib/
    ├── __init__.py
    ├── DoBotArm.py
    ├── DobotDllType.py
    └── DobotDll.dll
```

- **main_out.py**: Main script to run and test DoBot functions.
- **dobot_lib/**: Python package containing all DoBot-related code and the DLL.

---

## **Setup Instructions**

1. **Requirements**
   - Python 3.x
   - DoBot Magician hardware
   - All files (`DoBotArm.py`, `DobotDllType.py`, `DobotDll.dll`) must be inside the `dobot_lib` folder.

2. **DLL Placement**
   - Place `DobotDll.dll` in the `dobot_lib` folder (next to the Python files).

3. **Package Initialization**
   - Ensure `dobot_lib` contains an empty `__init__.py` file.

---

## **How to Run**

From the project root (`Conveyor belt pickup`), run:
```sh
python main_out.py
```

---

## **Key Features**

- **Portable DLL Loading:** The DLL is always loaded from the `dobot_lib` folder, regardless of where you run the script.
- **Package Imports:** Uses relative imports for robust module referencing.
- **Example Functions:** Includes sample functions for moving the arm, toggling suction, and manual control.

---

## **Troubleshooting**

- If you see `ModuleNotFoundError` or DLL loading errors, check that:
  - All files are in the correct folders as shown above.
  - You are running `python main_out.py` from the project root.
  - The DLL is named exactly `DobotDll.dll` and is in `dobot_lib`.

---

## **Extending**

- Add new DoBot control functions in `DoBotArm.py`.
- Use `main_out.py` as a template for your own scripts.

---
