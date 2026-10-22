Option Explicit
Dim shell, fs, folder, logs, batchFile, command, result
Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
folder = fs.GetParentFolderName(WScript.ScriptFullName)
logs = fs.BuildPath(folder, "logs")
If Not fs.FolderExists(logs) Then fs.CreateFolder logs
batchFile = fs.BuildPath(folder, "START_GREENPLAST.bat")
' cmd /s strips just the outer quotation pair. Paths remain quoted.
command = "cmd.exe /d /s /c " & Chr(34) & Chr(34) & batchFile & Chr(34) & _
          " --silent > " & Chr(34) & fs.BuildPath(logs, "windows_startup.log") & Chr(34) & _
          " 2>&1" & Chr(34)
result = shell.Run(command, 0, True)
If result <> 0 Then
    MsgBox "CV Waste Tools could not start (exit " & result & ")." & vbCrLf & _
           "See logs\windows_startup.log for details.", vbCritical, "CV Waste Tools"
End If
