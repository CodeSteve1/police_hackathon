from deep_translator import GoogleTranslator

text = "Hello, how are you?"
translated_text = GoogleTranslator(source="en", target="ta").translate(text)
print(translated_text)
