import os
import subprocess
import sys
import tempfile

from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File


class CompileSCSSPlugin(BasePlugin):
    def on_files(self, files, config):
        # Determine the path to your theme's SCSS file.
        # Since your theme lives in docs/theme, the SCSS file would be here:
        scss_file = os.path.join(config["docs_dir"], "theme", "scss", "theme.scss")
        if not os.path.exists(scss_file):
            sys.stdout.write(f"SCSS file not found: {scss_file}\n")
            return files

        # Use a temporary directory to store the compiled CSS.
        tempdir = tempfile.gettempdir()
        compiled_css = os.path.join(tempdir, "theme.css")

        sys.stdout.write(f"Compiling {scss_file} -> {compiled_css}...\n")
        try:
            # For example, using the pysassc command-line tool:
            subprocess.check_call(["pysassc", scss_file, compiled_css])
        except subprocess.CalledProcessError:
            sys.stdout.write("SCSS compilation failed.\n")
            return files

        # Define the destination directory inside your site output.
        # For instance, we want the CSS to be placed in site/assets/css/theme.css.
        dest_subdir = os.path.join("assets", "css")
        new_file = File(
            "theme.css",
            src_dir=tempdir,
            dest_dir=os.path.join(config["site_dir"], dest_subdir),
            use_directory_urls=False,
        )
        files.append(new_file)
        sys.stdout.write("SCSS compiled and added to site assets.\n")
        return files
