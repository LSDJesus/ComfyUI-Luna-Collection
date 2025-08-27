# SCRIPTURE: Interrogate-Directory.ps1
# PURPOSE: To force the machine to confess exactly what it sees in this directory.

$CurrentDirectory = $PSScriptRoot
Write-Host "--- INTERROGATION START ---" -ForegroundColor Yellow
Write-Host "The machine claims its current location is: $($CurrentDirectory)"
Write-Host ""

# --- INTERROGATION 1: ALL SOULS PRESENT ---
# We will first command the scout to report every single file it sees, with no filters.
# This will tell us if it is truly blind, or merely disobedient.
Write-Host "--- Confession 1: All files the scout can see ---" -ForegroundColor Cyan
Get-ChildItem -Path $CurrentDirectory | Format-Table Name, Extension, Length
Write-Host ""

# --- INTERROGATION 2: THE FLAWED COMMAND ---
# We will now run the exact command from the Chronicler and see what it reports.
Write-Host "--- Confession 2: Testing the Chronicler's exact command ---" -ForegroundColor Cyan
$ImageFiles = Get-ChildItem -Path "$($CurrentDirectory)\*" -Include @("*.png", "*.jpg", "*.jpeg", "*.webp")
Write-Host "The Chronicler's scout returned with $($ImageFiles.Count) souls."
Write-Host ""

# --- INTERROGATION 3: A SIMPLER TEST ---
# We will test a simpler, more brutal filter. This uses a different part of the scout's brain.
Write-Host "--- Confession 3: Testing a simpler filter for only PNG files ---" -ForegroundColor Cyan
$PngFiles = Get-ChildItem -Path $CurrentDirectory -Filter "*.png"
Write-Host "The simple scout returned with $($PngFiles.Count) PNG souls."
Write-Host ""

Write-Host "--- INTERROGATION END ---" -ForegroundColor Yellow