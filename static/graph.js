//JQUERY file for fetching graph data from html to the highcharts API
$(document).ready(function() {
	if (typeof chart_id !== 'undefined') {
		$(chart_id).highcharts({
			chart: chart,
			title: title,
			xAxis: xAxis,
			yAxis: yAxis,
			series: series,
			labels: labels
		});
	}
});

$(document).ready(function() {
	if (typeof chart2_id !== 'undefined') {
		$(chart2_id).highcharts({
			chart: chart2,
			title: title2,
			xAxis: xAxis2,
			yAxis: yAxis2,
			series: series2,
			labels: labels2,
			colorAxis: colorAxis2
		});
	}
});

$(document).ready(function() {
	if (typeof chart3_id !== 'undefined') {
		$(chart3_id).highcharts({
			chart: chart3,
			title: title3,
			xAxis: xAxis3,
			yAxis: yAxis3,
			series: series3,
			labels: labels3,
			plotOptions3: plotOptions3
		});
	}
});