' Windowless launcher for the durable L0 supervisor.
' Usage: wscript.exe run-hidden.vbs "<script.ps1>" [script arguments...]
Option Explicit

Dim shell, args, commandLine, i
Set shell = CreateObject("WScript.Shell")
Set args = WScript.Arguments

If args.Count = 0 Then
    WScript.Quit 1
End If

commandLine = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File " & Chr(34) & args(0) & Chr(34)
For i = 1 To args.Count - 1
    commandLine = commandLine & " " & Chr(34) & args(i) & Chr(34)
Next

' 0 = hidden window, False = do not wait for the child to exit.
shell.Run commandLine, 0, False
