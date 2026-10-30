param([int]$Root)
# Lower to BelowNormal every julia.exe whose ancestor chain contains $Root. Touches nothing else.
$all = Get-CimInstance Win32_Process
$byId = @{}; foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }
foreach ($p in $all | Where-Object Name -eq 'julia.exe') {
  $cur = $p; $hit = $false
  for ($i = 0; $i -lt 8 -and $cur; $i++) { if ([int]$cur.ParentProcessId -eq $Root) { $hit = $true; break }; $cur = $byId[[int]$cur.ParentProcessId] }
  if ($hit) { $g = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
    if ($g -and $g.PriorityClass -ne 'BelowNormal') { $g.PriorityClass = 'BelowNormal'; "lowered $($p.ProcessId)" } }
}
