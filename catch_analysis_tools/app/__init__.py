def create_app(test_config=None):
    # Backward-compatible factory that returns the Connexion-backed Flask app.
    from catch_analysis_tools.app.app import application

    return application
