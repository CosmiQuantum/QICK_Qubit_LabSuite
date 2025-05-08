# In Progress

---
## Fridge‐Specific JSON Parsers

Here lives parser classes that load fridge‐relative configuration files from outside the codebase. Each parser:

- Reads one or more `.json` files from a fridge’s config directory (e.g. `configs/QUIET/measurement_config.json`).  
- Validates and deserializes the JSON into Python objects for downstream use.  
- Keeps fridge settings decoupled from your main code, so you can add or update fridges simply by dropping new files into `configs/<FRIDGE_NAME>/`.
