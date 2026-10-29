Option Explicit
Dim shell, fs, folder, logs, batchFile, command, result
Dim latestLog, launchLog, stamp, errText, sink, newInstance
Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
' The normal shortcut restores an existing window. Only an explicit
' --new-instance shortcut bypasses that guard, for this launch only.
newInstance = False
If WScript.Arguments.Count > 0 Then
    If LCase(WScript.Arguments(0)) = "--new-instance" Then
        newInstance = True
    Else
        MsgBox "Unrecognized launcher option: " & WScript.Arguments(0), vbExclamation, "CV Waste Studio"
        WScript.Quit 2
    End If
End If
If newInstance Then shell.Environment("PROCESS")("CV_WASTE_STUDIO_NEW_INSTANCE") = "1"
folder = fs.GetParentFolderName(WScript.ScriptFullName)
logs = fs.BuildPath(folder, "logs")
If Not fs.FolderExists(logs) Then fs.CreateFolder logs
batchFile = fs.BuildPath(folder, "START_GREENPLAST.bat")

' Keep a separate log for each launch: simultaneous launches must not
' overwrite each other's diagnostics. Also keep windows_startup.log as the
' last-launch compatibility path.
Randomize
stamp = Year(Now) & Right("0" & Month(Now), 2) & Right("0" & Day(Now), 2) & _
        "_" & Right("0" & Hour(Now), 2) & Right("0" & Minute(Now), 2) & _
        Right("0" & Second(Now), 2) & "_" & Right("000000" & Int(Rnd * 1000000), 6)
launchLog = fs.BuildPath(logs, "windows_startup_" & stamp & ".log")
latestLog = fs.BuildPath(logs, "windows_startup.log")

' Catch a missing/unwritable log folder before invoking cmd.exe.
On Error Resume Next
Set sink = fs.CreateTextFile(launchLog, True)
If Err.Number <> 0 Then
    errText = Err.Description
    On Error GoTo 0
    MsgBox "Cannot create startup log in:" & vbCrLf & logs & vbCrLf & _
           errText, vbCritical, "CV Waste Studio"
    WScript.Quit 1
End If
sink.Close
On Error GoTo 0

' cmd /s strips just the outer quotation pair. Paths remain quoted.
command = "cmd.exe /d /s /c " & Chr(34) & Chr(34) & batchFile & Chr(34) & _
          " --silent > " & Chr(34) & launchLog & Chr(34) & _
          " 2>&1" & Chr(34)
On Error Resume Next
result = shell.Run(command, 0, True)
If Err.Number <> 0 Then
    errText = Err.Description
    Err.Clear
    result = 1
Else
    errText = ""
End If
' The dated log remains authoritative even if updating the latest copy fails.
fs.CopyFile launchLog, latestLog, True
Err.Clear
On Error GoTo 0
If result <> 0 Then
    MsgBox "CV Waste Studio could not start (exit " & result & ")." & vbCrLf & _
           IIfHelper(errText) & "Startup log:" & vbCrLf & launchLog, _
           vbCritical, "CV Waste Studio"
End If

Function IIfHelper(details)
    If details = "" Then
        IIfHelper = ""
    Else
        IIfHelper = details & vbCrLf
    End If
End Function
