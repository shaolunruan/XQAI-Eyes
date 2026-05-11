import axios from "axios";
import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
	PauseCircleFilled,
	PlayCircleFilled,
	RightOutlined,
} from "@ant-design/icons";
import {
	Button,
	Card,
	Col,
	Drawer,
	Form,
	InputNumber,
	Layout,
	Progress,
	Row,
	Slider,
} from "antd";
import Footer from "./Articles/footer";
import P1 from "./Articles/p1";
import Top from "./Articles/top";
import DataFlowLink from "./Components/data_flow_link";
import DataSelectorPanel from "./Components/data_selector_panel";
import DescriptionComp from "./Components/description";
import EncodedMapView from "./Components/encoded_map_view";
import EncoderStepMappingView from "./Components/encoder_step_mapping_view";
import ModelPerformanceView from "./Components/model_performance_view";
import OriginalDataView from "./Components/original_data_view";
import QuantumCircuitView from "./Components/quantum_circuit_view";
import QuantumStateDistributionView from "./Components/quantum_state_distribution_view";

import appConfig from "./configs.json";

const server_address =
	appConfig?.server_address ||
	process.env.REACT_APP_SERVER_ADDRESS ||
	"http://localhost:3030";

// 布局参数
const VIS_WIDTH = 1060;
const vis_height = VIS_WIDTH * 0.6;

const control_width = VIS_WIDTH;
const control_height = VIS_WIDTH * 0.1;

const original_data_view_width = VIS_WIDTH * 0.23;
const original_data_view_height = VIS_WIDTH * 0.26;
const original_data_view_left = VIS_WIDTH * 0.04,
	original_data_view_top = VIS_WIDTH * 0.023;

const data_selector_panel_width = VIS_WIDTH * 0.19;
const data_selector_panel_height = VIS_WIDTH * 0.133;
const data_selector_panel_left = VIS_WIDTH * 0.038,
	data_selector_panel_top = VIS_WIDTH * 0.34;

const quantum_circuit_width = VIS_WIDTH * 0.45;
const quantum_circuit_height = VIS_WIDTH * 0.07;
const quantum_circuit_left = VIS_WIDTH * 0.29,
	quantum_circuit_top = VIS_WIDTH * 0.023;

const encoded_map_width = VIS_WIDTH * 0.4;
const encoded_map_height = VIS_WIDTH * 0.295;
const encoded_map_left = VIS_WIDTH * 0.72,
	encoded_map_top = VIS_WIDTH * 0.023;

const data_flow_link_width =
	quantum_circuit_left -
	(original_data_view_left + original_data_view_width) +
	quantum_circuit_width +
	40;
const data_flow_link_height = VIS_WIDTH * 0.07;
const data_flow_link_left = original_data_view_width - 20,
	data_flow_link_top = quantum_circuit_top + quantum_circuit_height + 10;

const model_performance_view_width = VIS_WIDTH * 0.54;
const model_performance_view_height = VIS_WIDTH * 0.2;
const model_performance_view_left = VIS_WIDTH * 0.565,
	model_performance_view_top = VIS_WIDTH * 0.34;

const encoder_step_mapping_width = VIS_WIDTH * 0.45;
const encoder_step_mapping_height = VIS_WIDTH * 0.25;
const encoder_step_mapping_left = quantum_circuit_left,
	encoder_step_mapping_top = quantum_circuit_top + quantum_circuit_height + 10;

const quantum_state_distribution_view_width = VIS_WIDTH * 0.28;
const quantum_state_distribution_view_height = VIS_WIDTH * 0.145;
const quantum_state_distribution_view_left = VIS_WIDTH * 0.26,
	quantum_state_distribution_view_top = VIS_WIDTH * 0.34;

// Description: position it below the Original Data view
const description_panel_width = original_data_view_width;
const description_panel_height = original_data_view_height * 0.22;
const description_panel_left = original_data_view_left;
const description_panel_top =
	original_data_view_top + original_data_view_height + 6;

// Ensure components below description have enough padding
const below_desc_padding = 20;
const data_selector_panel_top_adjusted = Math.max(
	data_selector_panel_top,
	description_panel_top + description_panel_height + below_desc_padding,
);

const quantum_state_distribution_view_top_adjusted = Math.max(
	quantum_state_distribution_view_top,
	description_panel_top + description_panel_height + below_desc_padding,
);

const model_performance_view_top_adjusted = Math.max(
	model_performance_view_top,
	description_panel_top + description_panel_height + below_desc_padding,
);

const ARTICLE_WIDTH = 650;
const CENTERED_FOOTER_BG_COLOR = 650;

// All color setting here`
// define colors here
const COLOR_CLASS_1 = "#ffe682";
const COLOR_CLASS_2 = "#006962";
const TOP_BG_COLOR = "#183D4E";
const CENTERED_CONTROL_COLOR = "#ffffff";
const CONTAINER_CONTROL_COLOR = "#ffffff";
const CENTERED_VIS_COLOR = "#fafafa";
const CONTAINER_VIS_COLOR = "#fafafa";
const PLAY_BTN_COLOR = "#2c2c2c";
const PROGRESS_COLOR = "#545454";
const CENTERED_ARTICLE_BG_COLOR = "#ffffff";
const COLOR_DATA_SELECTOR_PANEL_BG = "#ececec";
const COLOR_CIRCUIT_BG = "#ececec";
const COLOR_ENCODED_MAP_BG = "#f1f1f1";
const COLOR_MODEL_PERFORMANCE_BG = "#f9f9f9";
const COLOR_ENCODER_STEP_MAP_BG = "#f9f9f9";
const COLOR_QUANTUM_DISTRIBUTION_BG = "#ececec";
const COLOR_LINK_BG = "#fafafa";

// default values
const DAFAULT_CIRCUIT = "circuit_5";
const DEFAULT_EPOCH = 100;
const DEFAULT_LR = 0.02;

// dataset and backend circuit ids
const DATA_PORT_MAP = {
	circuit_0: 0,
	circuit_1: 1,
	circuit_2: 2,
	circuit_3: 3,
	circuit_4: 4,
	circuit_5: 5,
};

function App() {
	const [data_name, set_dataName] = useState(DAFAULT_CIRCUIT);
	const [dataset, setDataset] = useState(null);
	const [encoders, setEncoders] = useState({});
	const [defaults, setDefaults] = useState({});
	const [selectedEncoder, setSelectedEncoder] = useState(null);
	const [epochNumber, setEpochNumber] = useState(DEFAULT_EPOCH);
	const [learningRate, setLearningRate] = useState(DEFAULT_LR);
	const [circuitPreview, setCircuitPreview] = useState(null);
	const [allData, setAllData] = useState(null);
	const [lastTrained, setLastTrained] = useState({
		circuitId: null,
		encoderName: null,
		epochNumber: null,
		learningRate: null,
	});

	// Cross-view selection (index in original dataset)
	const [selectedDataIndex, setSelectedDataIndex] = useState(null);

	// Streaming training session state
	const [sessionId, setSessionId] = useState(null);
	const eventSourceRef = useRef(null);
	const [trainingActive, setTrainingActive] = useState(false);
	const [paused, setPaused] = useState(false);
	const [currentEpoch, setCurrentEpoch] = useState(0);

	const [drawer_open, set_drawer_open] = useState(false);
	const prevDataNameRef = useRef(data_name);
	const initialFetchDoneRef = useRef(false);

	const handleDatasetClick = (datasetName) => {
		// Clear encoder first to avoid transient fetch with previous encoder
		setSelectedEncoder(null);
		set_dataName(datasetName);
	};

	/*设置抽屉的状态的函数*/
	const showDrawer = () => {
		set_drawer_open(true);
	};

	const onClose = () => {
		set_drawer_open(false);
	};

	// Helper: cleanup current EventSource
	const cleanupStream = async (doStop = false) => {
		try {
			if (doStop && sessionId) {
				await axios.post(`${server_address}/api/train/stop`, {
					session_id: sessionId,
				});
			}
		} catch (e) {}
		if (eventSourceRef.current) {
			try {
				eventSourceRef.current.close();
			} catch (e) {}
			eventSourceRef.current = null;
		}
		setSessionId(null);
		setTrainingActive(false);
		setPaused(false);
		setCurrentEpoch(0);
	};

	// Helper: whether current selection matches last trained configuration
	const isCurrentSelectionTrained = () => {
		const currentCircuitId = DATA_PORT_MAP[data_name];
		if (!lastTrained) return false;
		return (
			lastTrained.circuitId === currentCircuitId &&
			lastTrained.encoderName === selectedEncoder &&
			lastTrained.epochNumber === epochNumber &&
			lastTrained.learningRate === learningRate
		);
	};

	// Load encoders once
	useEffect(() => {
		const fetchEncoders = async () => {
			try {
				const res = await axios.get(`${server_address}/api/get_encoders`);
				setEncoders(res.data.encoders || {});
				setDefaults(res.data.defaults || {});
			} catch (err) {
				console.error("Failed to fetch encoders", err);
			}
		};
		fetchEncoders();
	}, []);

	// Sync selected encoder with dataset defaults when circuit changes
	useEffect(() => {
		const circuit_id = DATA_PORT_MAP[data_name];
		if (defaults && Object.hasOwn(defaults, circuit_id)) {
			setSelectedEncoder(defaults[circuit_id]);
		}
	}, [data_name, defaults]);

	// One-time initial fetch on first mount after defaults/encoder are ready
	useEffect(() => {
		const defaultsReady = defaults && Object.keys(defaults).length > 0;
		if (!defaultsReady) return;
		if (!selectedEncoder) return;
		if (initialFetchDoneRef.current) return;
		const circuit_id = DATA_PORT_MAP[data_name];
		const request_url = `${server_address}/api/run_circuit`;
		const payload = {
			circuit: circuit_id,
			encoder_name: selectedEncoder,
			epoch_number: epochNumber,
			lr: learningRate,
		};
		const fetchData = async () => {
			try {
				setDataset(null);
				prevDataNameRef.current = data_name;
				const result = await axios.post(request_url, payload);
				initialFetchDoneRef.current = true;
				setDataset(result.data);
				setLastTrained({
					circuitId: circuit_id,
					encoderName: selectedEncoder,
					epochNumber: epochNumber,
					learningRate: learningRate,
				});
			} catch (err) {
				console.error("Failed to load initial dataset", err);
				initialFetchDoneRef.current = true;
				setDataset(null);
			}
		};
		fetchData();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [defaults, selectedEncoder]);

	// Unmount cleanup
	useEffect(() => {
		return () => {
			cleanupStream(false);
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	// Start streaming training
	const startStreamingTraining = async () => {
		const circuit_id = DATA_PORT_MAP[data_name];
		const start_url = `${server_address}/api/train/start`;
		const payload = {
			circuit: circuit_id,
			epoch_number: epochNumber,
			lr: learningRate,
		};
		if (selectedEncoder) payload.encoder_name = selectedEncoder;
		try {
			await cleanupStream(true);
			setDataset(null);
			prevDataNameRef.current = data_name;
			const res = await axios.post(start_url, payload);
			const newSessionId = res.data.session_id;
			setSessionId(newSessionId);
			setTrainingActive(true);
			setPaused(false);
			setCurrentEpoch(0);
			// Initialize dataset holder for progressive updates
			setDataset({
				performance: { epoch_number: epochNumber, loss: [], accuracy: [] },
				trained_data: { feature: [], label: [] },
				distribution_map: [],
				feature_map_formula: encoders[selectedEncoder]?.feature_map_formula,
				circuit: circuitPreview || null,
			});
			setLastTrained({
				circuitId: circuit_id,
				encoderName: selectedEncoder,
				epochNumber: epochNumber,
				learningRate: learningRate,
			});

			// Open SSE stream
			const streamUrl = `${server_address}/api/train/stream?session_id=${newSessionId}`;
			const es = new EventSource(streamUrl);
			eventSourceRef.current = es;
			es.onmessage = (ev) => {
				try {
					const msg = JSON.parse(ev.data);
					if (msg.type === "session") {
						return;
					}
					if (msg.type === "epoch") {
						setCurrentEpoch(msg.epoch || 0);
						setDataset((prev) => {
							const perf = prev?.performance || {
								epoch_number: msg.epoch_number,
								loss: [],
								accuracy: [],
							};
							const nextLoss = Array.isArray(perf.loss)
								? [...perf.loss, msg.loss]
								: [msg.loss];
							const nextAcc = Array.isArray(perf.accuracy)
								? [...perf.accuracy, msg.accuracy]
								: [msg.accuracy];
							return {
								...prev,
								performance: {
									epoch_number: msg.epoch_number,
									loss: nextLoss,
									accuracy: nextAcc,
								},
								trained_data: msg.trained_data,
								distribution_map: msg.distribution_map,
								feature_map_formula:
									msg.feature_map_formula || prev?.feature_map_formula,
							};
						});
						return;
					}
					if (msg.type === "done") {
						setTrainingActive(false);
						setPaused(false);
						if (eventSourceRef.current) {
							eventSourceRef.current.close();
							eventSourceRef.current = null;
						}
						setSessionId(null);
						return;
					}
				} catch (e) {
					console.error("Failed to parse SSE message", e);
				}
			};
			es.onerror = () => {
				if (eventSourceRef.current) {
					eventSourceRef.current.close();
					eventSourceRef.current = null;
				}
				setTrainingActive(false);
				setPaused(false);
				setSessionId(null);
			};
		} catch (err) {
			console.error("Failed to start streaming training", err);
			setTrainingActive(false);
			setPaused(false);
			setSessionId(null);
		}
	};

	const togglePauseResume = async () => {
		if (!sessionId) return;
		try {
			if (!paused) {
				await axios.post(`${server_address}/api/train/pause`, {
					session_id: sessionId,
				});
				setPaused(true);
			} else {
				await axios.post(`${server_address}/api/train/resume`, {
					session_id: sessionId,
				});
				setPaused(false);
			}
		} catch (e) {
			console.error("Failed to toggle pause/resume", e);
		}
	};

	const stopTraining = async () => {
		await cleanupStream(true);
	};

	// Update circuit preview when encoder or dataset changes
	useEffect(() => {
		if (
			selectedEncoder &&
			encoders &&
			Object.hasOwn(encoders, selectedEncoder)
		) {
			const steps = encoders[selectedEncoder].steps || [];
			const stepsCount = Array.isArray(steps) ? steps.length : 0;
			const preview = {
				qubit_number:
					(dataset && dataset.circuit && dataset.circuit.qubit_number) || 2,
				encoder_step: stepsCount,
				encoder: steps,
				ansatz: (dataset && dataset.circuit && dataset.circuit.ansatz) || [],
				measure: (dataset && dataset.circuit && dataset.circuit.measure) || [],
			};
			setCircuitPreview(preview);
		} else {
			setCircuitPreview(null);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [selectedEncoder, encoders]);

	// Fetch encoded data when dataset selection or encoder changes
	useEffect(() => {
		const circuit_id = DATA_PORT_MAP[data_name];
		if (!selectedEncoder) {
			setAllData(null);
			return;
		}
		const source = axios.CancelToken.source();
		axios
			.get(`${server_address}/api/get_data`, {
				params: { circuit_id, encoder_name: selectedEncoder },
				cancelToken: source.token,
			})
			.then((res) => {
				const d = res.data || {};
				const features = d.original_features || [];
				const labels = d.original_labels || [];
				const combine = (labels) => ({
					feature: features,
					label: labels || [],
				});
				const encoded = d.encoded_data ? combine(d.encoded_data.label) : null;
				const steps = Array.isArray(d.encoded_steps)
					? d.encoded_steps.map((s) => combine(s.label))
					: null;
				const stepsSub = Array.isArray(d.encoded_steps_sub)
					? d.encoded_steps_sub.map((pair) => [
							combine(pair && pair[0] ? pair[0].label : []),
							combine(pair && pair[1] ? pair[1].label : []),
						])
					: null;
				setAllData({
					encoded_data: encoded,
					encoded_steps: steps,
					encoded_steps_sub: stepsSub,
					distribution_map: d.distribution_map || null,
					original_features: features,
					original_labels: labels,
				});
			})
			.catch((err) => {
				if (!axios.isCancel(err)) {
					console.error("Failed to fetch encoded data quickly", err);
				}
				setAllData(null);
			});
		return () => {
			source.cancel("Route changed");
		};
	}, [data_name, selectedEncoder, encoders]);

	return (
		<Layout
			style={{
				height: "100%",
				width: "100%",
				backgroundColor: "#ffffff" /* *** */,
				position: "relative",
			}}
		>
			{/* the top board component */}
			<Top top_bg_color={TOP_BG_COLOR}></Top>

			{/* GUI for vis interface */}
			<div
				className={"container-for-centered"}
				style={{ backgroundColor: CONTAINER_CONTROL_COLOR /* *** */ }}
			>
				<div
					id={"control"}
					className={"centered"}
					style={{
						height: control_height,
						width: control_width,
						backgroundColor: CENTERED_CONTROL_COLOR /* *** */,
						padding: "1.5em",
					}}
				>
					<div style={{ marginTop: "10px", marginLeft: "1.4em" }}>
						<span className={"control_font"}>Specify encoder</span>
						<Row
							style={{ width: "260px", marginTop: "5px", alignItems: "center" }}
						>
							<Col span={18}>
								<span
									onClick={!trainingActive ? showDrawer : undefined}
									style={{
										cursor: trainingActive ? "not-allowed" : "pointer",
										color: "#1677ff",
										opacity: trainingActive ? 0.6 : 1,
									}}
								>
									{selectedEncoder || "(default)"}
								</span>
							</Col>
							<Col>
								<Button
									style={{ width: "100%", marginLeft: "0.2em", height: "32px" }}
									onClick={showDrawer}
									size={"small"}
									disabled={trainingActive}
								>
									<RightOutlined />
								</Button>
								<Drawer
									title={null}
									placement="right"
									closable={true}
									onClose={onClose}
									open={drawer_open}
									width={640}
									bodyStyle={{ padding: 16 }}
								>
									<div
										style={{
											fontSize: 18,
											fontWeight: 600,
											marginBottom: 12,
											opacity: trainingActive ? 0.6 : 1,
										}}
									>
										Select an encoder
									</div>
									<Row gutter={[12, 12]}>
										{Object.entries(encoders).map(([name, encoder]) => {
											const steps = encoder.steps;
											const stepsCount = Array.isArray(steps)
												? steps.length
												: 0;
											const isSelected = selectedEncoder === name;
											const previewCircuit = {
												qubit_number:
													(dataset &&
														dataset.circuit &&
														dataset.circuit.qubit_number) ||
													2,
												encoder_step: stepsCount,
												encoder: steps,
												ansatz:
													(dataset &&
														dataset.circuit &&
														dataset.circuit.ansatz) ||
													[],
												measure:
													(dataset &&
														dataset.circuit &&
														dataset.circuit.measure) ||
													[],
											};
											return (
												<Col xs={24} sm={12} key={name}>
													<Card
														hoverable
														onClick={() => {
															if (trainingActive) return;
															setSelectedEncoder(name);
															set_drawer_open(false);
														}}
														style={{
															width: "100%",
															border: isSelected
																? "2px solid #1677ff"
																: undefined,
															opacity: trainingActive ? 0.6 : 1,
															cursor: trainingActive
																? "not-allowed"
																: "pointer",
														}}
													>
														<div
															style={{
																display: "flex",
																justifyContent: "space-between",
																alignItems: "center",
															}}
														>
															<div style={{ fontWeight: 600 }}>{name}</div>
														</div>
														<QuantumCircuitView
															id={"enc-preview"}
															style={{ opacity: trainingActive ? 0.6 : 1 }}
															dataset={previewCircuit}
															width={280}
															height={90}
															left={0}
															top={0}
															color_circuit_bg={COLOR_CIRCUIT_BG}
															svgId={`enc-card-${name}`}
															forCards={true}
														/>
													</Card>
												</Col>
											);
										})}
									</Row>
								</Drawer>
							</Col>
						</Row>
					</div>

					<div style={{ marginTop: "10px", marginRight: "1.5em" }}>
						<span className={"control_font"}>Training epoch</span>
						<Progress
							percent={
								trainingActive
									? Math.min(
											100,
											Math.round(
												(currentEpoch / Math.max(1, epochNumber)) * 100,
											),
										)
									: 0
							}
							status="active"
							strokeColor={PROGRESS_COLOR}
							style={{ width: "150px", marginRight: "-30px", marginTop: "5px" }}
						/>
					</div>

					<div className="button-group">
						<Button
							className={"play-button"}
							style={{
								color: PLAY_BTN_COLOR,
								marginLeft: "2em",
								marginRight: "2em",
							}}
							icon={
								trainingActive ? (
									paused ? (
										<PlayCircleFilled style={{ fontSize: "4.5em" }} />
									) : (
										<PauseCircleFilled style={{ fontSize: "4.5em" }} />
									)
								) : (
									<PlayCircleFilled style={{ fontSize: "4.5em" }} />
								)
							}
							type="text"
							onClick={
								trainingActive ? togglePauseResume : startStreamingTraining
							}
						/>
						<Button
							icon={
								<span className="material-icons" style={{ fontSize: "2em" }}>
									replay
								</span>
							}
							type="text"
							onClick={async () => {
								if (trainingActive) {
									await stopTraining();
								} else {
									setEpochNumber(DEFAULT_EPOCH);
									setLearningRate(DEFAULT_LR);
								}
							}}
						/>
					</div>

					<div style={{ marginTop: "10px" }}>
						<span className={"control_font"}>Epoch number</span>
						<Form.Item
							style={{ width: "200px", marginRight: "-30px", marginTop: "5px" }}
						>
							<Row>
								<Col span={14}>
									<Slider
										min={1}
										max={1000}
										step={1}
										value={epochNumber}
										onChange={setEpochNumber}
										disabled={trainingActive}
									/>
								</Col>
								<Col>
									<InputNumber
										style={{ width: "60px" }}
										size={"small"}
										min={1}
										max={1000}
										value={epochNumber}
										onChange={(v) => {
											if (typeof v === "number") setEpochNumber(v);
										}}
										disabled={trainingActive}
									/>
								</Col>
							</Row>
						</Form.Item>
					</div>

					<div style={{ marginTop: "10px", marginRight: "1em" }}>
						<span className={"control_font"}>Learning rate</span>
						<Form.Item
							style={{ width: "200px", marginRight: "-30px", marginTop: "5px" }}
						>
							<Row>
								<Col span={14}>
									<Slider
										min={0}
										max={0.1}
										step={0.01}
										value={learningRate}
										onChange={setLearningRate}
										disabled={trainingActive}
									/>
								</Col>
								<Col>
									<InputNumber
										style={{ width: "60px" }}
										size={"small"}
										min={0}
										max={0.1}
										step={0.01}
										value={learningRate}
										onChange={(v) => {
											if (typeof v === "number") setLearningRate(v);
										}}
										disabled={trainingActive}
									/>
								</Col>
							</Row>
						</Form.Item>
					</div>
				</div>
			</div>

			{/* vis interface*/}
			<div
				className={"container-for-centered"}
				style={{ backgroundColor: CONTAINER_VIS_COLOR /* *** */ }}
			>
				<div
					id={"vis_system"}
					style={{
						height: vis_height,
						width: VIS_WIDTH,
						backgroundColor: CENTERED_VIS_COLOR /* *** */,
					}}
				>
					{/* Component-1: original data view*/}
					<OriginalDataView
						features={allData?.original_features}
						labels={allData?.original_labels}
						selectedIndex={selectedDataIndex}
						onHoverIndex={setSelectedDataIndex}
						class_color={[COLOR_CLASS_1, COLOR_CLASS_2]}
						width={original_data_view_width}
						height={original_data_view_height}
						left={original_data_view_left}
						top={original_data_view_top}
					></OriginalDataView>

					{/* Component-2: data selector panel (always rendered) */}
					<DataSelectorPanel
						default_circuit={data_name}
						onDatasetClick={handleDatasetClick}
						colors={[
							[COLOR_CLASS_1, COLOR_CLASS_2],
							COLOR_DATA_SELECTOR_PANEL_BG,
						]}
						width={data_selector_panel_width}
						height={data_selector_panel_height}
						left={data_selector_panel_left}
						top={data_selector_panel_top_adjusted}
						vis_width={VIS_WIDTH}
						disabled={trainingActive}
					></DataSelectorPanel>

					{/* Component-3: quantum circuit show*/}
					{circuitPreview && (
						<QuantumCircuitView
							key={`${data_name}-${selectedEncoder || "default"}`}
							dataset={circuitPreview}
							width={quantum_circuit_width}
							height={quantum_circuit_height}
							left={quantum_circuit_left}
							top={quantum_circuit_top}
							color_circuit_bg={COLOR_CIRCUIT_BG}
						></QuantumCircuitView>
					)}

					{/* Component-4: encoded map*/}
					{allData?.encoded_data && (
						<EncodedMapView
							dataset={allData.encoded_data}
							circuit_id={DATA_PORT_MAP[data_name]}
							colors={[[COLOR_CLASS_1, COLOR_CLASS_2], COLOR_ENCODED_MAP_BG]}
							width={encoded_map_width}
							height={encoded_map_height}
							left={encoded_map_left}
							top={encoded_map_top}
							selectedIndex={selectedDataIndex}
							onHoverIndex={setSelectedDataIndex}
						></EncodedMapView>
					)}

					{/* Link: animated line from Comp1 to Comp4*/}
					{allData?.encoded_data && (
						<DataFlowLink
							boundary={null}
							colors={[[COLOR_CLASS_1, COLOR_CLASS_2], COLOR_LINK_BG]}
							linkComp_width={data_flow_link_width}
							linkComp_height={data_flow_link_height}
							linkComp_left={data_flow_link_left}
							linkComp_top={data_flow_link_top}
						></DataFlowLink>
					)}

					{/* Component-5: Model performance view*/}
					{dataset && isCurrentSelectionTrained() && (
						<ModelPerformanceView
							dataset1={dataset["performance"]}
							dataset2={dataset["trained_data"]}
							colors={[
								[COLOR_CLASS_1, COLOR_CLASS_2],
								COLOR_MODEL_PERFORMANCE_BG,
							]}
							width={model_performance_view_width}
							height={model_performance_view_height}
							left={model_performance_view_left}
							top={model_performance_view_top_adjusted}
							selectedIndex={selectedDataIndex}
							onHoverIndex={setSelectedDataIndex}
						></ModelPerformanceView>
					)}

					{/* Component-6: encoder step map*/}
					{allData?.encoded_steps && allData?.encoded_steps_sub && (
						<EncoderStepMappingView
							key={`${data_name}-${selectedEncoder || "default"}-steps`}
							dataset={[allData.encoded_steps, allData.encoded_steps_sub]}
							width={encoder_step_mapping_width}
							height={encoder_step_mapping_height}
							left={encoder_step_mapping_left}
							top={encoder_step_mapping_top}
							colors={[
								[COLOR_CLASS_1, COLOR_CLASS_2],
								COLOR_ENCODER_STEP_MAP_BG,
							]}
						></EncoderStepMappingView>
					)}

					{/* Component-7: Quantum state distribution*/}
					{allData?.distribution_map && (
						<QuantumStateDistributionView
							dataset={allData.distribution_map}
							width={quantum_state_distribution_view_width}
							height={quantum_state_distribution_view_height}
							left={quantum_state_distribution_view_left}
							top={quantum_state_distribution_view_top_adjusted}
							colors={[
								[COLOR_CLASS_1, COLOR_CLASS_2],
								COLOR_QUANTUM_DISTRIBUTION_BG,
							]}
							selectedIndex={selectedDataIndex}
							onHoverIndex={setSelectedDataIndex}
						></QuantumStateDistributionView>
					)}

					{/* Component-descriptionComp: some description*/}
					{encoders && encoders[selectedEncoder] && (
						<DescriptionComp
							formula={encoders[selectedEncoder].feature_map_formula}
							left={description_panel_left}
							top={description_panel_top}
							width={description_panel_width}
							height={description_panel_height}
						/>
					)}
				</div>
			</div>

			<article>
				<div
					className="article-body"
					style={{
						width: ARTICLE_WIDTH,
						background: CENTERED_ARTICLE_BG_COLOR,
					}}
				>
					<P1></P1> {/*Article here*/}
				</div>
			</article>

			<footer>
				<div
					className="centered-footer"
					style={{ width: ARTICLE_WIDTH, background: CENTERED_FOOTER_BG_COLOR }}
				>
					<Footer></Footer>
				</div>
			</footer>
		</Layout>
	);
}

export default App;
