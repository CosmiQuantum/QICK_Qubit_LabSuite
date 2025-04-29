import visdom

def check_visdom_connection(live_plot=False):
    if live_plot:
        # Check if visdom is connected right away, otherwise, throw an error
        if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                               "http://localhost:8097/ on firefox")