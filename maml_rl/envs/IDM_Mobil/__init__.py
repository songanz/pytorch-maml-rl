# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# Import the envs module so that envs register themselves
import maml_rl.envs.IDM_Mobil.envs
