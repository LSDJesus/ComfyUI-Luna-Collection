# SCRIPTURE: Forge-Chronicle.ps1 (v1.1)
# PURPOSE: To find every image and its family of .txt files and forge them into a single, magnificent Markdown dossier.

# This is the path to the directory where the script is currently running.
$CurrentDirectory = $PSScriptRoot

# This is the name of the grand chronicle we will create.
$ChronicleFile = Join-Path -Path $CurrentDirectory -ChildPath "_Chronicle.md"

# --- The Forging Begins ---

# We begin with a clean slate, writing the title of our grand work.
Set-Content -Path $ChronicleFile -Value "# The Pyrite Judgments: A Grand Chronicle`n"

Write-Host "The forge is hot. I will now begin the chronicle..."

# The scout's first duty: find every unique image (.png, .jpg, etc.) to serve as our subject.
# The corrected path ensures the scout sees all the trees in the forest.
$ImageFiles = Get-ChildItem -Path "$($CurrentDirectory)" -Include @("*.png", "*.jpg", "*.jpeg", "*.webp") | Sort-Object Name

# A diagnostic line to ensure the scout is working.
Write-Host "Scout has returned with $($ImageFiles.Count) souls to chronicle."

# The commander's duty: for each image, compile its complete dossier.
foreach ($Image in $ImageFiles) {
    $BaseName = $Image.BaseName
    
    # This is the list of scriptures we will hunt for each subject.
    $ScriptureSuffixes = @(
        "_tag.txt",
        "_tag_caption_1.txt",
        "_tag_caption_2.txt",
        "_tag_caption_3.txt"
    )

    # Begin the scripture for this single soul.
    $DossierEntry = "## $($Image.Name)`n`n"
    $DossierEntry += "![Image]($($Image.Name))`n`n"

    # Now, the scribe will hunt for each piece of scripture.
    foreach ($Suffix in $ScriptureSuffixes) {
        $TextFilePath = Join-Path -Path $CurrentDirectory -ChildPath "$($BaseName)$($Suffix)"
        
        # We add the heading for this piece of scripture.
        $DossierEntry += "### $($Suffix.Replace('.txt',''))`n"
        
        if (Test-Path $TextFilePath) {
            # If the scroll exists, we transcribe its contents into a sacred code block.
            $FileContent = Get-Content -Path $TextFilePath -Raw
            $DossierEntry += "````text`n$($FileContent)`n`````n`n"
        } else {
            # If the scroll is missing, we note its absence. The chronicle must be complete.
            $DossierEntry += "*-- Scripture not found. --*`n`n"
        }
    }

    # We close the dossier for this soul with a final, clean line.
    $DossierEntry += "`n---\n"
    
    # The scribe adds this completed entry to the Grand Chronicle.
    Add-Content -Path $ChronicleFile -Value $DossierEntry
    Write-Host "Chronicled: $($Image.Name)"
}

Write-Host "The ritual is complete. The Grand Chronicle has been written to: $($ChronicleFile)"