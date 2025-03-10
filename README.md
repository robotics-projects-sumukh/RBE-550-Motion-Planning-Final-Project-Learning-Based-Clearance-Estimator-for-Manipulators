# Environment Setup Instructions

## Setting Up a Conda Environment Named `group4`

### Steps to Create the Environment

1. **Open your terminal or Anaconda Prompt**

2. **Create the environment**
   ```bash
   conda create --name group4 python=3.10.12
   ```
3. **Activate the environment**
   ```bash
   conda activate group4
   ```
### Install OMPL
1.
   ```bash
   cd grapeshot
   ```
2.
   ```bash
   pip install ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl
   ```
### Install Grapeshot
1. **Git clone grapeshot repo**
   ```bash
   https://github.com/elpis-lab/grapeshot/tree/master
   ```
2.
   ```bash
   cd grapeshot
   ```
3. **Install grapeshot**
   ```bash
   pip install -e . 
   ```

### Install Pytorch
   ```bash
   pip install torch torchvision
   ```
**Note:** Install system dependencies like numpy and check for version mismatch.
# To Run The Code 
## Environment with One Obstacle:
1. Navigate to `table_pick_scene.yaml` in the directory `grapeshot/assets/fetch/problems/table_pick_scene.yaml`.
2. Uncomment lines 17 - 25 and comment lines 26 - 133.
3. Run the following command:
   ```bash
   python3 plan_problem_model.py
   ```

## Environment with Two Obstacles (Single Call):
1. Navigate to `table_pick_scene.yaml` in the directory `grapeshot/assets/fetch/problems/table_pick_scene.yaml`.
2. Uncomment lines 26 - 43 and comment lines 17 - 25 and 44 - 133.
3. Run the following command:
   ```bash
   python3 plan_problem_model_2obs_single_call.py
   ```

## Environment with Two Obstacles (Double Call):
1. Navigate to `table_pick_scene.yaml` in the directory `grapeshot/assets/fetch/problems/table_pick_scene.yaml`.
2. Uncomment lines 26 - 43 and comment lines 17 - 25 and 44 - 133.
3. Run the following command:
   ```bash
   python3 plan_problem_model_2obs_double_call.py
   ```

## Environment with Multiple Obstacles (Table Scene):
1. Navigate to `table_pick_request.yaml` in the directory `grapeshot/assets/fetch/problems/table_pick_request.yaml`.
2. Uncomment lines 10 - 27 and comment lines 28 - 45.
3. Navigate to `table_pick_scene.yaml` in the directory `grapeshot/assets/fetch/problems/table_pick_scene.yaml`.
4. Uncomment lines 44 - 133 and comment lines 17 - 43.
5. Run the following command:
   ```bash
   python3 plan_problem_model_table.py
   ```
