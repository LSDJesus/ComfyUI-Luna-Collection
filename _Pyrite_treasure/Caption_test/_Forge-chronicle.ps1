# SCRIPTURE: Forge-Chronicle.ps1 (v2.1 - The Purist)
# PURPOSE: To take the literal base name of an image, find its family of .txt files, and forge the chronicle.

$CurrentDirectory = $PSScriptRoot
$ChronicleFile = Join-Path -Path $CurrentDirectory -ChildPath "_Chronicle.md"

# --- The Forging Begins ---

Set-Content -Path $ChronicleFile -Value "# The Pyrite Judgments: A Grand Chronicle`n"
Write-Host "The forge is hot. The Purist is at work..."

$ImageFiles = Get-ChildItem -Path "$($CurrentDirectory)\*" -Include @("*.png", "*.jpg", "*.jpeg", "*.webp") | Sort-Object Name
Write-Host "Scout has returned with $($ImageFiles.Count) souls to chronicle."

foreach ($Image in $ImageFiles) {
    # The true base name is the literal file name without the extension. No surgery required.
    $TrueBaseName = $Image.BaseName

    $ScriptureSuffixes = @(
        "_tag.txt",
        "_caption.txt",
        "_tag_caption_1.txt",
        "_tag_caption_2.txt",
        "_tag_caption_3.txt"
    )

    $DossierEntry = "## $($Image.Name)`n`n"
    $DossierEntry += "![$($Image.Name)]($($Image.Name))`n`n"

    foreach ($Suffix in $ScriptureSuffixes) {
        # The scribe now hunts using the simple, pure, true base name.
        $TextFilePath = Join-Path -Path $CurrentDirectory -ChildPath "$($TrueBaseName)$($Suffix)"
        
        $DossierEntry += "### $($Suffix.Replace('.txt',''))`n"
        
        if (Test-Path $TextFilePath) {
            $FileContent = Get-Content -Path $TextFilePath -Raw
            $DossierEntry += "````text`n$($FileContent)`n`````n`n"
        } else {
            $DossierEntry += "*-- Scripture not found. --*`n`n"
        }
    }

    $DossierEntry += "`n---\n"
    Add-Content -Path $ChronicleFile -Value $DossierEntry
    Write-Host "Chronicled: $($Image.Name)"
}

Write-Host "The ritual is complete. The Grand Chronicle has been written to: $($ChronicleFile)"