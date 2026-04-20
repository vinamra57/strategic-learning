# strategic-learning

## python3 model.py --agent-type pg --no-critic --n-episodes 50000 --log-every 1000 --eval-every 5000 --eval-episodes 200$

## python3 model.py --agent-type pg --no-critic --no-sit-out --n-episodes 25000 --log-every 1000 --eval-every 5000 --eval-episodes 200

pg --n-episodes=25000 no-critic pg-coeff=0.05 lr=1e-3 ~38% winrate
pg --n-episodes=25000 with critic pg-coeff=0.05 lr=1e-3 ~42% winrate
pg --n-episodes=25000 no-critic pg-coeff=0.05 lr=1e-4 ~47.5% winrate