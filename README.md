# zero_shot_object_detection
### GUI for ViLD, like just a very simple one.

.

├── README.md

├── examples

│  ├── test.jpg

├── icon

│  ├── icons8-assessments-90.png

│  └── icons8-multiplication-90.png

├── image_path_v2

│  ├── saved_model.pb

│  └── variables

│        ├── variables.data-00000-of-00001

│        └── variables.index

├── result

│  └── result_1.jpg

├── tmp

│  ├── ui.jpg

└── vild_ui.ipynb

To save the memory, README is simple.

| **examples/**  | the .jpg file as the original dataset |
| -------------- | ------------------------------------- |
| **icon/**      | Yes or No icon (in Qt)                |
| image_path_v2/ | Just as the tpu repo                  |
| **result/**    | image saved by UI                     |
| **tmp/**       | used by UI (in the Qt process)        |
| Vild_ui.ipynb  | Jupiter file to run                   |

I have tried to `pip freeze` to get the requirements.txt
