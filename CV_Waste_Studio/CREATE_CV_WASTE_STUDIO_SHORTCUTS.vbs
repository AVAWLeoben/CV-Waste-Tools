' Make Windows shortcuts that show our icon instead of the generic .vbs symbol.
Option Explicit
Dim shell, fs, folder, iconFile, launcherFile, winScript, desktopFolder
Dim folderError, desktopError, resultMessage, createdCount
Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
folder = fs.GetParentFolderName(WScript.ScriptFullName)
iconFile = fs.BuildPath(folder, "assets\cv_waste_studio_windows_v5.ico")
launcherFile = fs.BuildPath(folder, "START_CV_WASTE_TOOLS.vbs")
winScript = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\wscript.exe")

If Not fs.FileExists(iconFile) Or Not fs.FileExists(launcherFile) Then
    MsgBox "The icon or startup script is missing. Extract the entire ZIP, including the assets folder.", vbCritical, "CV Waste Studio"
    WScript.Quit 1
End If

Function MakeShortcut(destination)
    Dim link
    On Error Resume Next
    Err.Clear
    ' Remove the existing .lnk first; changing IconLocation alone may leave a stale Explorer icon.
    If fs.FileExists(destination) Then
        fs.DeleteFile destination, True
    End If
    If Err.Number <> 0 Then
        MakeShortcut = Err.Description
        On Error GoTo 0
        Exit Function
    End If
    Set link = shell.CreateShortcut(destination)
    link.TargetPath = winScript
    link.Arguments = Chr(34) & launcherFile & Chr(34)
    link.WorkingDirectory = folder
    link.IconLocation = iconFile & ",0"
    link.Description = "CV Waste Studio - model training, auditing and evaluation"
    link.WindowStyle = 1
    link.Save
    If Err.Number <> 0 Then
        MakeShortcut = Err.Description
    Else
        MakeShortcut = ""
    End If
    On Error GoTo 0
End Function

folderError = MakeShortcut(fs.BuildPath(folder, "CV Waste Studio.lnk"))
On Error Resume Next
desktopFolder = shell.SpecialFolders("Desktop")
If Err.Number <> 0 Then
    desktopError = Err.Description
    Err.Clear
Else
    desktopError = MakeShortcut(fs.BuildPath(desktopFolder, "CV Waste Studio.lnk"))
End If
On Error GoTo 0

createdCount = 0
resultMessage = ""
If folderError = "" Then
    createdCount = createdCount + 1
    resultMessage = "Application folder: created" & vbCrLf
Else
    resultMessage = "Application folder: " & folderError & vbCrLf
End If
If desktopError = "" Then
    createdCount = createdCount + 1
    resultMessage = resultMessage & "Desktop: created" & vbCrLf
Else
    resultMessage = resultMessage & "Desktop: " & desktopError & vbCrLf
End If
If createdCount > 0 Then
    MsgBox "CV Waste Studio shortcuts:" & vbCrLf & resultMessage & vbCrLf & _
           "Use the freshly created CV Waste Studio shortcut, not the .vbs file." & vbCrLf & _
           "If an old taskbar icon remains, unpin it and pin the running Studio again.", vbInformation, "CV Waste Studio"
Else
    MsgBox "No shortcuts could be created:" & vbCrLf & resultMessage, vbCritical, "CV Waste Studio"
    WScript.Quit 1
End If
