"""Internationalization support for global deployment."""


class InternationalizationManager:
    """Manages international language support."""
    
    def __init__(self):
        self.supported_languages = ["en", "es", "fr", "de", "ja", "zh"]
    
    def get_supported_languages(self):
        return self.supported_languages