package workflow

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/hhcho/sfgwas/mpc"
)

type stageDefinition struct {
	parent          string
	measurementKind string
}

// Caps follow the flattened chromosome/gene order used by the embedding pipeline.
func capCountTable(chromosomes []ChromosomeInput, caps []int) string {
	header := []string{"M"}
	totals := make([]int, len(chromosomes)+1)
	counts := make(map[int][]int)
	gene := 0
	for c, chromosome := range chromosomes {
		header = append(header, fmt.Sprintf("chr%d", chromosome.Chromosome))
		for range chromosome.Genes {
			m := caps[gene]
			if counts[m] == nil {
				counts[m] = make([]int, len(totals))
			}
			counts[m][c]++
			counts[m][len(chromosomes)]++
			gene++
		}
		totals[c] = len(chromosome.Genes)
	}
	totals[len(chromosomes)] = gene
	header = append(header, "Total")
	sizes := make([]int, 0, len(counts))
	for m := range counts {
		sizes = append(sizes, m)
	}
	sort.Ints(sizes)
	rows := [][]string{header}
	addRow := func(label string, values []int) {
		row := []string{label}
		for _, value := range values {
			row = append(row, strconv.Itoa(value))
		}
		rows = append(rows, row)
	}
	for _, m := range sizes {
		addRow(strconv.Itoa(m), counts[m])
	}
	addRow("Total", totals)
	return formatMetricTable(rows, true)
}

func formatMetricTable(rows [][]string, totalRow bool) string {
	widths := make([]int, len(rows[0]))
	for _, row := range rows {
		for c, cell := range row {
			widths[c] = max(widths[c], len(cell))
		}
	}
	var table strings.Builder
	for r, row := range rows {
		if r == 1 || (totalRow && r == len(rows)-1) {
			line := make([]string, len(widths))
			for c, width := range widths {
				line[c] = strings.Repeat("─", width)
			}
			fmt.Fprintln(&table, strings.Join(line, "  "))
		}
		for c, cell := range row {
			if c > 0 {
				table.WriteString("  ")
			}
			fmt.Fprintf(&table, "%*s", widths[c], cell)
		}
		table.WriteByte('\n')
	}
	return table.String()
}

// Compare elapsed seconds across caps without summing parent and child timers.
// The last column normalizes for the number of table chunks; site steps are
// summed within this party only, and include communication/collective waits.
func (recorder *metricRecorder) coordinateTimeTable() string {
	var sizes []int
	parents := make(map[int]metricEvent)
	for _, event := range recorder.events {
		if strings.HasPrefix(event.stage, "embedding_M") && strings.HasSuffix(event.stage, "_coordinates") {
			m, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(event.stage, "embedding_M"), "_coordinates"))
			sizes = append(sizes, m)
			parents[m] = event
		}
	}
	if len(sizes) == 0 {
		return ""
	}
	sort.Ints(sizes)
	rows := [][]string{{"M", "Chunks", "Prepare", "Anchor", "Lookup", "Assign", "Update", "Transfer", "Other", "Total", "s/chunk"}}
	for _, m := range sizes {
		parent := parents[m]
		sums := make([]time.Duration, 7)
		for _, event := range recorder.events {
			step, ok := strings.CutPrefix(event.stage, parent.stage+"_")
			if !ok {
				continue
			}
			column := 6
			switch {
			case step == "prepare":
				column = 0
			case step == "anchor":
				column = 1
			case strings.HasSuffix(step, "_lookup"):
				column = 2
			case strings.HasSuffix(step, "_assign"):
				column = 3
			case strings.HasSuffix(step, "_update"):
				column = 4
			case strings.HasSuffix(step, "_transfer"), strings.HasSuffix(step, "_broadcast"):
				column = 5
			}
			sums[column] += event.duration
		}
		known := time.Duration(0)
		for _, duration := range sums[:6] {
			known += duration
		}
		sums[6] = max(0, parent.duration-known)
		row := []string{strconv.Itoa(m), strconv.Itoa(parent.count)}
		for _, duration := range sums {
			row = append(row, fmt.Sprintf("%.3f", duration.Seconds()))
		}
		row = append(row, fmt.Sprintf("%.3f", parent.duration.Seconds()), fmt.Sprintf("%.3f", parent.duration.Seconds()/float64(parent.count)))
		rows = append(rows, row)
	}
	return formatMetricTable(rows, false)
}

var stageDefinitions = map[string]stageDefinition{
	"input_loading":         {measurementKind: "leaf"},
	"network_init":          {parent: "session_setup", measurementKind: "leaf"},
	"sample_count_exchange": {parent: "session_setup", measurementKind: "leaf"},
	"alignment":             {measurementKind: "inclusive"}, // all chromosomes, before the statistics
	"collective_setup":      {parent: "session_setup", measurementKind: "inclusive"},
	"pubkey_gen":            {parent: "collective_setup", measurementKind: "leaf"},
	"relin_key_gen":         {parent: "collective_setup", measurementKind: "leaf"},
	"rotkey_gen":            {parent: "collective_setup", measurementKind: "leaf"},
	"null_model":            {measurementKind: "inclusive"},
	"null_local_equations":  {parent: "null_model", measurementKind: "leaf"},
	"null_aggregate_shares": {parent: "null_model", measurementKind: "leaf"},
	"null_factor_solve":     {parent: "null_model", measurementKind: "leaf"},
	"null_rss":              {parent: "null_model", measurementKind: "leaf"},
	"chromosome_total":      {measurementKind: "inclusive"},
	"compute_weights":       {parent: "chromosome_total", measurementKind: "leaf"},
	"dosage_loading":        {parent: "chromosome_total", measurementKind: "leaf"},
	"local_dosage_sums":     {parent: "chromosome_total", measurementKind: "leaf"},
	"packed_statistics":     {parent: "chromosome_total", measurementKind: "inclusive"},
	"genotype_loading":      {parent: "packed_statistics", measurementKind: "leaf"},
	"beta_packing":          {parent: "packed_statistics", measurementKind: "leaf"},
	"batch_preparation":     {parent: "packed_statistics", measurementKind: "inclusive"},
	"local_gtx":             {parent: "batch_preparation", measurementKind: "leaf"},
	"local_gtg":             {parent: "batch_preparation", measurementKind: "leaf"},
	"public_ql":             {parent: "packed_statistics", measurementKind: "inclusive"},
	"local_gty":             {parent: "public_ql", measurementKind: "leaf"},
	"kernel_inputs":         {parent: "packed_statistics", measurementKind: "leaf"},
	"first_gtg_action":      {parent: "packed_statistics", measurementKind: "leaf"},
	"second_gtg_action":     {parent: "packed_statistics", measurementKind: "leaf"},
	"burden_variance":       {parent: "packed_statistics", measurementKind: "leaf"},
	"finalize":              {parent: "chromosome_total", measurementKind: "inclusive"},
	"alpha_score_assembly":  {parent: "finalize", measurementKind: "leaf"},
	"burden_statistic":      {parent: "finalize", measurementKind: "leaf"},
	"skat_wilson_hilferty":  {parent: "finalize", measurementKind: "leaf"},
	"release":               {parent: "chromosome_total", measurementKind: "inclusive"},
	"open_pivots":           {parent: "release", measurementKind: "leaf"},
	"local_pvalues":         {parent: "release", measurementKind: "leaf"},
	"write_results":         {measurementKind: "leaf"},
}

type metricEvent struct {
	stage         string
	chromosome    int
	duration      time.Duration
	sentBytes     uint64
	receivedBytes uint64
	count         int
}

type processMetric struct {
	process      string
	duration     time.Duration
	peakRSSBytes uint64
}

type packedWidthMetric struct {
	chromosome      int
	width           int
	probeCount      int
	geneCount       int
	ciphertextCount int
	duration        time.Duration
}

type metricRecorder struct {
	eventsMu       sync.Mutex
	packedWidthsMu sync.Mutex
	process        string
	ancestry       string
	printEvents    bool
	startedAt      time.Time
	events         []metricEvent
	packedWidths   []packedWidthMetric
}

func newMetricRecorder(
	process string,
	ancestry string,
	printEvents bool,
) *metricRecorder {
	return &metricRecorder{
		process:     process,
		ancestry:    ancestry,
		printEvents: printEvents,
		startedAt:   time.Now(),
	}
}

func (recorder *metricRecorder) start(
	stage string,
	chromosome int,
	networks mpc.ParallelNetworks,
) func() {
	return recorder.startEvent(stage, chromosome, networks, true)
}

func (recorder *metricRecorder) observe(
	chromosome int,
	networks mpc.ParallelNetworks,
) func(stage string) func() {
	return func(stage string) func() {
		return recorder.startEvent(stage, chromosome, networks, false)
	}
}

func (recorder *metricRecorder) observePackedWidth(
	chromosome int,
	probeCount int,
) func(width, geneCount int) func() {
	return func(width, geneCount int) func() {
		startedAt := time.Now()
		return func() {
			metric := packedWidthMetric{
				chromosome:      chromosome,
				width:           width,
				probeCount:      probeCount,
				geneCount:       geneCount,
				ciphertextCount: 1,
				duration:        time.Since(startedAt),
			}

			recorder.packedWidthsMu.Lock()
			defer recorder.packedWidthsMu.Unlock()

			for index := range recorder.packedWidths {
				current := &recorder.packedWidths[index]
				if current.chromosome != chromosome || current.width != width {
					continue
				}
				current.geneCount += metric.geneCount
				current.ciphertextCount += metric.ciphertextCount
				current.duration += metric.duration
				return
			}

			recorder.packedWidths = append(recorder.packedWidths, metric)
		}
	}
}

func (recorder *metricRecorder) startEvent(
	stage string,
	chromosome int,
	networks mpc.ParallelNetworks,
	printEvent bool,
) func() {
	startedAt := time.Now()
	var startCommunication mpc.CommunicationStats
	if len(networks) > 0 {
		startCommunication = networks.GetCommunicationStats()
	}

	return func() {
		var sentBytes, receivedBytes uint64
		if len(networks) > 0 {
			communication := networks.GetCommunicationStats().Sub(
				startCommunication,
			)
			sentBytes = communication.SentBytes
			receivedBytes = communication.ReceivedBytes
		}

		event := metricEvent{
			stage:         stage,
			chromosome:    chromosome,
			duration:      time.Since(startedAt),
			sentBytes:     sentBytes,
			receivedBytes: receivedBytes,
			count:         1,
		}
		recorder.addEvent(event)

		if !recorder.printEvents || !printEvent {
			return
		}
		if event.chromosome == 0 {
			fmt.Printf(
				"[%s %s] %s: %.3fs\n",
				recorder.process,
				recorder.ancestry,
				event.stage,
				event.duration.Seconds(),
			)
		} else {
			fmt.Printf(
				"[%s %s] chr%d %s: %.3fs\n",
				recorder.process,
				recorder.ancestry,
				event.chromosome,
				event.stage,
				event.duration.Seconds(),
			)
		}
	}
}

func (recorder *metricRecorder) addDuration(
	stage string,
	chromosome int,
	duration time.Duration,
) {
	recorder.addEvent(metricEvent{
		stage:      stage,
		chromosome: chromosome,
		duration:   duration,
		count:      1,
	})
}

func (recorder *metricRecorder) addEvent(event metricEvent) {
	recorder.eventsMu.Lock()
	defer recorder.eventsMu.Unlock()
	for index := range recorder.events {
		current := &recorder.events[index]
		if current.stage != event.stage || current.chromosome != event.chromosome {
			continue
		}

		current.duration += event.duration
		current.sentBytes += event.sentBytes
		current.receivedBytes += event.receivedBytes
		current.count += event.count
		return
	}
	recorder.events = append(recorder.events, event)
}

func (recorder *metricRecorder) writeCSV(path string) error {
	rows := make([][]string, 0, len(recorder.events))
	for _, event := range recorder.events {
		definition := stageDefinitions[event.stage]
		if strings.HasPrefix(event.stage, "embedding_") {
			definition = stageDefinition{parent: "alignment", measurementKind: "leaf"}
			if prefix, _, detail := strings.Cut(event.stage, "_coordinates_"); detail {
				definition.parent = prefix + "_coordinates"
			} else if strings.HasSuffix(event.stage, "_coordinates") {
				definition.measurementKind = "inclusive"
			}
		}
		chromosome := ""
		if event.chromosome != 0 {
			chromosome = strconv.Itoa(event.chromosome)
		}
		rows = append(rows, []string{
			recorder.process,
			recorder.ancestry,
			event.stage,
			definition.parent,
			definition.measurementKind,
			chromosome,
			strconv.Itoa(event.count),
			strconv.FormatFloat(
				event.duration.Seconds(),
				'f',
				9,
				64,
			),
			strconv.FormatUint(event.sentBytes, 10),
			strconv.FormatUint(event.receivedBytes, 10),
		})
	}

	return writeCSV(path, []string{
		"process",
		"ancestry",
		"stage",
		"parent_stage",
		"measurement_kind",
		"chromosome",
		"count",
		"duration_seconds",
		"sent_bytes",
		"received_bytes",
	}, rows)
}

func (recorder *metricRecorder) printTimeTree() {
	if !recorder.printEvents {
		return
	}
	fmt.Print(recorder.timeTree(time.Since(recorder.startedAt)))
}

func (recorder *metricRecorder) timeTree(total time.Duration) string {
	stageDuration := func(stage string, chromosome int) time.Duration {
		for _, event := range recorder.events {
			if event.stage == stage && event.chromosome == chromosome {
				return event.duration
			}
		}
		return 0
	}
	format := func(duration time.Duration) time.Duration {
		return duration.Round(time.Millisecond)
	}

	stageTotal := func(stage string) time.Duration { // summed over every chromosome
		var sum time.Duration
		for _, event := range recorder.events {
			if event.stage == stage {
				sum += event.duration
			}
		}
		return sum
	}
	setup := stageDuration("network_init", 0) +
		stageDuration("sample_count_exchange", 0) +
		stageDuration("collective_setup", 0)
	alignmentTotal := stageTotal("alignment")
	inputLoading := stageDuration("input_loading", 0)
	nullModel := stageDuration("null_model", 0)
	writeResults := stageDuration("write_results", 0)
	classified := inputLoading + setup + alignmentTotal + nullModel + writeResults

	var tree strings.Builder
	fmt.Fprintf(
		&tree,
		"\n[%s %s] timing tree:\n",
		recorder.process,
		recorder.ancestry,
	)
	fmt.Fprintf(&tree, "  ├─ input loading          %v\n", format(inputLoading))
	fmt.Fprintf(&tree, "  ├─ session setup          %v\n", format(setup))
	fmt.Fprintf(&tree, "  │  ├─ network init         %v\n", format(stageDuration("network_init", 0)))
	fmt.Fprintf(&tree, "  │  ├─ sample count exchange %v\n", format(stageDuration("sample_count_exchange", 0)))
	fmt.Fprintf(&tree, "  │  ├─ collective setup     %v\n", format(stageDuration("collective_setup", 0)))
	fmt.Fprintf(&tree, "  │  │  ├─ PubKeyGen          %v\n", format(stageDuration("pubkey_gen", 0)))
	fmt.Fprintf(&tree, "  │  │  ├─ RelinKeyGen        %v\n", format(stageDuration("relin_key_gen", 0)))
	fmt.Fprintf(&tree, "  │  │  └─ RotKeyGen          %v\n", format(stageDuration("rotkey_gen", 0)))
	fmt.Fprintf(&tree, "  ├─ cap + MHE embedding    %v\n", format(alignmentTotal))
	fmt.Fprintf(&tree, "  │  ├─ private cap          %v\n", format(stageTotal("embedding_private_cap")))
	fmt.Fprintf(&tree, "  │  ├─ embedding keygen     %v\n", format(stageTotal("embedding_keygen")))
	for _, event := range recorder.events {
		if strings.HasPrefix(event.stage, "embedding_M") && !strings.Contains(event.stage, "_coordinates_") {
			fmt.Fprintf(&tree, "  │  ├─ %-17s %v (%d chunks)\n", strings.TrimPrefix(event.stage, "embedding_"), format(event.duration), event.count)
		}
	}
	fmt.Fprintf(&tree, "  ├─ null model             %v\n", format(nullModel))
	fmt.Fprintf(&tree, "  │  ├─ local XTX/XTY/YTY    %v\n", format(stageDuration("null_local_equations", 0)))
	fmt.Fprintf(&tree, "  │  ├─ aggregate shares     %v\n", format(stageDuration("null_aggregate_shares", 0)))
	fmt.Fprintf(&tree, "  │  ├─ factor and solve     %v\n", format(stageDuration("null_factor_solve", 0)))
	fmt.Fprintf(&tree, "  │  └─ RSS                  %v\n", format(stageDuration("null_rss", 0)))

	for _, event := range recorder.events {
		if event.stage != "chromosome_total" {
			continue
		}
		chromosome := event.chromosome
		classified += event.duration
		fmt.Fprintf(&tree, "  ├─ chr%-19d %v\n", chromosome, format(event.duration))
		fmt.Fprintf(&tree, "  │  ├─ weights              %v\n", format(stageDuration("compute_weights", chromosome)))
		fmt.Fprintf(&tree, "  │  ├─ packed statistics    %v\n", format(stageDuration("packed_statistics", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ beta packing       %v\n", format(stageDuration("beta_packing", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ batch preparation  %v\n", format(stageDuration("batch_preparation", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ public Q/L         %v\n", format(stageDuration("public_ql", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ kernel inputs      %v\n", format(stageDuration("kernel_inputs", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ first GtG action   %v\n", format(stageDuration("first_gtg_action", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ second GtG action  %v\n", format(stageDuration("second_gtg_action", chromosome)))
		fmt.Fprintf(&tree, "  │  │  └─ burden variance    %v\n", format(stageDuration("burden_variance", chromosome)))
		fmt.Fprintf(&tree, "  │  ├─ finalize             %v\n", format(stageDuration("finalize", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ alpha and score assembly %v\n", format(stageDuration("alpha_score_assembly", chromosome)))
		fmt.Fprintf(&tree, "  │  │  ├─ burden statistic   %v\n", format(stageDuration("burden_statistic", chromosome)))
		fmt.Fprintf(&tree, "  │  │  └─ SKAT Wilson-Hilferty %v\n", format(stageDuration("skat_wilson_hilferty", chromosome)))
		fmt.Fprintf(&tree, "  │  └─ release              %v\n", format(stageDuration("release", chromosome)))
		fmt.Fprintf(&tree, "  │     ├─ open pivots        %v\n", format(stageDuration("open_pivots", chromosome)))
		fmt.Fprintf(&tree, "  │     └─ local p-values     %v\n", format(stageDuration("local_pvalues", chromosome)))
	}

	other := total - classified
	if other < 0 {
		other = 0
	}
	fmt.Fprintf(&tree, "  ├─ write results          %v\n", format(writeResults))
	fmt.Fprintf(&tree, "  ├─ other overhead         %v\n", format(other))
	fmt.Fprintf(&tree, "  └─ TOTAL                  %v\n", format(total))
	if table := recorder.coordinateTimeTable(); table != "" {
		fmt.Fprintf(&tree, "\n[%s %s] coordinate steps (elapsed seconds, includes waits):\n%s", recorder.process, recorder.ancestry, table)
	}

	for _, chromosomeEvent := range recorder.events {
		if chromosomeEvent.stage != "chromosome_total" {
			continue
		}
		chromosome := chromosomeEvent.chromosome
		fmt.Fprintf(
			&tree,
			"\n[%s %s] chr%d packed statistics:\n",
			recorder.process,
			recorder.ancestry,
			chromosome,
		)
		fmt.Fprintf(&tree, "  W      mode       genes   ctxt  duration(s)\n")
		for _, metric := range recorder.packedWidths {
			if metric.chromosome != chromosome {
				continue
			}
			mode := "exact"
			if metric.width > metric.probeCount {
				mode = "hutch"
			}
			fmt.Fprintf(
				&tree,
				"  %-6d %-7s %8d %6d %12.3f\n",
				metric.width,
				mode,
				metric.geneCount,
				metric.ciphertextCount,
				metric.duration.Seconds(),
			)
		}
	}
	return tree.String()
}

func writeProcessMetricsCSV(path string, metrics []processMetric) error {
	rows := make([][]string, 0, len(metrics))
	for _, metric := range metrics {
		rows = append(rows, []string{
			metric.process,
			strconv.FormatFloat(metric.duration.Seconds(), 'f', 9, 64),
			strconv.FormatUint(metric.peakRSSBytes, 10),
		})
	}

	return writeCSV(path, []string{
		"process",
		"duration_seconds",
		"peak_rss_bytes",
	}, rows)
}

func writeCSV(path string, header []string, rows [][]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create CSV directory: %w", err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create CSV file: %w", err)
	}

	writer := csv.NewWriter(file)
	writer.WriteAll(append([][]string{header}, rows...))
	if err := writer.Error(); err != nil {
		file.Close()
		return fmt.Errorf("write CSV file: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close CSV file: %w", err)
	}
	return nil
}

func peakRSSBytes(usage *syscall.Rusage) (uint64, error) {
	peakRSS := uint64(usage.Maxrss)
	switch runtime.GOOS {
	case "darwin":
		return peakRSS, nil
	case "linux":
		return peakRSS * 1024, nil
	default:
		return 0, fmt.Errorf("peak RSS is not supported on %s", runtime.GOOS)
	}
}

func currentProcessPeakRSSBytes() (uint64, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, fmt.Errorf("read process resource usage: %w", err)
	}
	return peakRSSBytes(&usage)
}

func childProcessPeakRSSBytes(state *os.ProcessState) (uint64, error) {
	if state == nil {
		return 0, fmt.Errorf("child process state is unavailable")
	}
	usage, ok := state.SysUsage().(*syscall.Rusage)
	if !ok {
		return 0, fmt.Errorf("child process resource usage is unavailable")
	}
	return peakRSSBytes(usage)
}
