## current_version

Each row is stored in the following format, and the latest version is always written in the first row.

```
[version]\t[newly added models which is separated by commas]\t"[update messages]"
```

e.g. The contents of `current_version` are as follows:

```
3 hyaudio_1,hyaudio_2 "add hyaudio_1 and hyaudio_2 model"
// others
```

That means the latest model version is `3`, the newly added models is `hyaudio_1` and `hyaudio_2`,
and the updata message is `add hyaudio_1 and hyaudio_2 model`