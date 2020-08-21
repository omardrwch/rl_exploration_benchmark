from rlxp.rendering.render2d import Render2D 

def render_env2d(env):
    """
    Function to render an environment that implements RenderInterface2D
    """
    if env.is_render_enabled():
        # background 
        background = env.get_background()

        # data: convert states to scenes
        data = []
        for state in env._state_history_for_rendering:
            scene = env.get_scene(state)
            data.append(scene)
        
        # render 
        renderer = Render2D() 
        renderer.set_refresh_interval(env._refresh_interval)
        renderer.set_clipping_area(env._clipping_area)
        renderer.set_data(data)
        renderer.set_background(background)
        renderer.run_graphics() 
        return 0
    else:
        print("Rendering not enabled for the environment.")
        return 1