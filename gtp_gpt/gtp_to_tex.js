// Convert a song from "gp/gp*" format to a "tex" flat text format.
// Takes two arguments, e.g. `node tex_to_gtp.js score.gp score.tex`

var t = require("@coderline/alphatab_7f82ec0a");
const alphatab = t.alphaTab;
const fs = require('fs');

var args = process.argv.slice(2);
const inPath = args[0];
const outPath = args[1];

var inBuf = fs.readFileSync(inPath);
// var bb = alphatab.io.ByteBuffer.FromBuffer(inBuf);
// const readerBase = new alphatab.importer.Gp7Importer();
// var gi = new alphatab.importer.GpxImporter();
// gi.Init(bb, new alphatab.Settings());
// let score = gi.ReadScore();
var score = alphatab.importer.ScoreLoader.LoadScoreFromBytes(inBuf, new alphatab.Settings());

// find first bass track
let track = ''
for (let i = 0; i < score.Tracks.length; i++) { 
	if (score.Tracks[i].PlaybackInfo.Program >= 32 && score.Tracks[i].PlaybackInfo.Program <= 40) {
		track = score.Tracks[i]
	} 
}

let exporter = new alphatab.exporter.AlphaTexExporter();
exporter.Export(track);
fs.writeFileSync(outPath, exporter.ToTex(), 'utf8');
