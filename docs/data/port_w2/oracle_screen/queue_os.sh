#!/bin/zsh
O=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/oscreen
$O/run_os.sh X >> $O/queue_os.out 2>&1 || exit 1
$O/run_os.sh T >> $O/queue_os.out 2>&1
echo OSDONE >> $O/queue_os.out
