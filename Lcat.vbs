Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Set environment
WshShell.Environment("Process")("PATH") = appDir & "\python;" & appDir & "\runtime\torch\lib;" & appDir & "\runtime\av.libs;" & appDir & "\runtime\PyQt5\Qt5\bin;" & appDir & "\runtime\cv2;" & appDir & ";" & WshShell.Environment("Process")("PATH")
WshShell.Environment("Process")("PYTHONPATH") = appDir & "\src;" & appDir & "\runtime"

' Launch without console
WshShell.Run """" & appDir & "\python\pythonw.exe"" """ & appDir & "\src\main.py""", 0, False
