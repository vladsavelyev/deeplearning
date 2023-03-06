### Using transformers for guitar tablature.

Tablature (tabs) is a common notation to represent guitar music. Unlike sheet music, tabs is easier to read for beginners as it directly shows which guitar fret needs to be played, e.g. the notation below shows that the 7th fret of the 4th string needs to be played twice, followed by the 5th fret on the 3rd string, followed by an open 2nd string, and finally 4th string and the first string are played at the same time.

G|-7-7-------7-|
D|-----5-5-----|
A|---------0---|
E|-----------7-|

We attempt to represent it in a flat string suitable for training with a transformer. Perhaps a CNN network would be natural here as well as long as the structure of music is known in advance (e.g. 4 strings and 4 beats in a measure might sit well in a 2d convolution). Though that structure is not guaranteed to be constant during the song, and in my opinion song is very like the natural text, and needs the attention mechanism to find self-references, etc.

