launchd job that emails Hannah's (Entry A) picks Tuesday and Thursday at 8:30am.
Missed runs (laptop asleep) fire at next wake.

Install / update:

    cp launchd/com.sbruton.pickem-email.plist ~/Library/LaunchAgents/
    launchctl bootout gui/$(id -u)/com.sbruton.pickem-email 2>/dev/null
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.sbruton.pickem-email.plist

Run now (test):    launchctl kickstart -k gui/$(id -u)/com.sbruton.pickem-email
Log:               picks/email.log
Config:            config.json (gitignored; copy config.example.json)
