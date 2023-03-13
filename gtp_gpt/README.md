### Using transformers for guitar tablature.

Tablature (tabs) is a common notation to represent guitar music. Unlike sheet music, tabs is easier to read for beginners as it directly shows which guitar fret needs to be played, e.g. the notation below shows that the 7th fret of the 4th string needs to be played twice, followed by the 5th fret on the 3rd string, followed by an open 2nd string, and finally 4th string and the first string are played at the same time.

G|-7-7-------7-|
D|-----5-5-----|
A|---------0---|
E|-----------7-|


Transformer-based NLP models feel natural to be applied to music, as music is similar in text, having a structure with self-references and rhymes, suitable for the attention mechanism. So first we want to find a way to convert tabs into a flat text format, suitable for tokenization. 

### `alphaTex` format

The [alphaTex](https://alphatab.net/docs/alphatex/introduction/#:~:text=AlphaTex%20is%20a%20text%20format,the%20features%20alphaTab%20supports%20overall.) format is introduced by the [alphaTab](https://github.com/CoderLine/alphaTab) tool, open source platform that supports tools to work with tabs. We can clone the repo and install it as a node module in order to use its features to work with alphaTex.

```sh
git submodule add http://github.com/CoderLine/alphaTab.git
cd alphaTab
npm run build
cd ..
rm -rf node_modules/@coderline/alphatab/dist
cp -r alphaTab/dist node_modules/@coderline/alphatab/
```

Unfortunately, there is only a `AlphaTexImporter` modudle in the project. We digged a bit and found that the `AlphaTexExporter` module existed, but starting from [this commit](https://github.com/CoderLine/alphaTab/tree/a15680687214b4f9d85832a4152e98f4feeb5590), when the project was rewritten to TypeScript, it was removed. The last commit to have `AlphaTexExporter` was `7f82ec0aa36bbb6d7cea57785202563f677ac859`. We can just use that revision instead of `main` We actually go a bit further down the history and use a bit earlier commit, that had pre-built JS artefacts, so we don't have to rebuild ourselves. We wget the artefact directly from the repo:

```sh
cp -r node_modules/@coderline/alphatab node_modules/@coderline/alphatab_7f82ec0a
# get the compiled JS version (last commit it was released):
wget https://github.com/CoderLine/alphaTab/raw/fd29edaf872834de50612005adcecf4a1c9597be/Build/JavaScript/AlphaTab.js \
	-O node_modules/@coderline/alphatab_7f82ec0a/dist/alphaTab.js
```

Now, to conver gp files to tex and reverse, we implemented two scripts:

```sh
node gtp_to_tex.js test/metallica.gp4 test/out-metallica.tex
node tex_to_gtp.js test/metallica.tex test/out-metallica.gp7
```