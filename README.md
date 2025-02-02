# Setup Instructions

## Install Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```


To run base DQN:
```
python main.py --env MinAtar/Breakout-v1
```

To run multistep extension for DQN:
```
python main.py --env MinAtar/Breakout-v1 --multistep --n_steps 3
```
