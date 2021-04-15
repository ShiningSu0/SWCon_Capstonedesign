import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Portfolio-v0',
    entry_point='gym_portfolio.envs:PortfolioEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)
