"""각 지표의 정의와 출처를 구분한 공개 명세. 특정 합격 점수는 정하지 않는다."""
# 지표 용어·단위·계산 정의·근거 문헌을 정리한 사전입니다.
# 수식의 출처와 프로젝트 자체 파라미터를 구분합니다. 논문이 80%나 95%를 합격점으로 정해준 것은 아닙니다.
# 전체 임무 전용 키의 보완 정의는 tools/finalize_full_results.py와 docs/metric_catalog.csv에 있습니다.

REFERENCES=[
 {"id":"S01","title":"ITU R P 525 5 Calculation of free space attenuation","url":"https://www.itu.int/rec/R-REC-P.525/en","location":"자유공간 기본 전송 손실","use":"거리·주파수 손실 수식","evidence":"공식 표준 문서"},
 {"id":"S02","title":"ITU R P 526 16 Propagation by diffraction","url":"https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.526-16-202511-I%21%21PDF-E.pdf","location":"4.5.1 식 49~57, 표준 본문 28~29쪽","use":"평지 Bullington 부분을 건물 지붕 프로파일에 적용","evidence":"공식 PDF 전체 확보 후 해당 수식 확인. 4.5.2 complete method는 구현하지 않음"},
 {"id":"S03","title":"MathWorks Analytical Expressions Used in berawgn","url":"https://www.mathworks.com/help/comm/ug/analytical-expressions-used-in-berawgn-function-and-bit-error-rate-analysis-app.html","location":"M PSK 중 M=2 BPSK 구간","use":"coherent uncoded BPSK의 이론 BER","evidence":"공식 방법 문서. 패킷 길이 변환과 간섭의 AWGN 치환은 구현 가정"},
 {"id":"S04","title":"De Couto et al 2003 A High Throughput Path Metric for Multi Hop Wireless Routing","url":"https://pdos.csail.mit.edu/~decouto/my-papers/mobicom03.pdf","location":"3 The ETX Metric","use":"전달·ACK 성공률의 역수라는 경로 비용 근거","evidence":"저자 기관 공개 PDF. 실측 probe ETX나 DSDV DSR를 재현한 구현은 아님"},
 {"id":"S05","title":"RFC 7680 A One Way Loss Metric for IP Performance Metrics","url":"https://www.rfc-editor.org/rfc/rfc7680.html","location":"2.4, 2.8.2, 4.1","use":"유실 판정 시한과 전달률 분모 명시","evidence":"표준 본문. 본 구현은 IP 스택이 아닌 추상 패킷"},
 {"id":"S06","title":"RFC 7679 A One Way Delay Metric for IP Performance Metrics","url":"https://www.rfc-editor.org/rfc/rfc7679.html","location":"2.4, 2.8.2","use":"생성·수신 시각의 차이 및 유실 패킷 지연을 0으로 처리하지 않는 원칙","evidence":"표준 본문. 0.25초·2초는 자체 실험 설정"},
 {"id":"S07","title":"Latora and Marchiori 2001 Efficient Behavior of Small World Networks","url":"https://www.w3.org/People/Massimo/papers/2001/efficiency_prl_01.pdf","location":"네트워크 효율 정의","use":"도달 불가능한 쌍의 효율을 0으로 둔 평균 역최단거리","evidence":"저자 공개 PDF. 여기서는 가중 물리 거리 대신 홉 수"},
 {"id":"S08","title":"Jain Chiu and Hawe 1984 A Quantitative Measure Of Fairness","url":"https://www.cs.wustl.edu/~jain/papers/fairness.htm","location":"Jain fairness index","use":"흐름별 PDR와 드론별 관측 셀 수의 균형","evidence":"저자 기관 원 보고서. 적용 변수는 본 과제에서 선택"},
 {"id":"S09","title":"Fan Chung Spectral Graph Theory","url":"https://fanchung.ucsd.edu/research/outline.html","location":"Chapter 1 Eigenvalues and the Laplacian of a graph","use":"정규화 라플라시안과 두 번째 고유값","evidence":"저자 공개 교재 안내. 연결성의 합격 점수를 제공하는 자료는 아님"},
 {"id":"S10","title":"Yates et al Age of Information An Introduction and Survey","url":"https://arxiv.org/abs/2007.08564","location":"AoI 기본 정의","use":"현재 시각에서 최신 수신 상태 패킷 생성 시각을 뺀 값","evidence":"원 논문. 본 구현은 프레임 끝의 샘플 평균, 초기 생성 시각 0 가정"},
 {"id":"S11","title":"Yanmaz et al 2024 Dynamic Multi UAV Path Planning for Multi Target Search and Connectivity","url":"https://doi.org/10.1109/TVT.2024.3363840","location":"출판사 공개 초록","use":"수색과 전용 중계를 나누어 통신·수색의 상충 관계를 검토할 배경","evidence":"이번 구현 근거 확인은 초록 범위. 원 논문은 GCS가 있으므로 본 과제와 시나리오가 같지 않음"},
 {"id":"S12","title":"Multi UAV Coverage Path Planning in Urban Environments Sensors 2021 21 7365","url":"https://www.mdpi.com/1424-8220/21/21/7365","location":"도시 장애물 환경의 다중 UAV 커버리지 경로 계획","use":"장애물 회피와 대형 변형을 함께 관찰할 배경","evidence":"출판사 공개 본문. 원 논문의 FM2 방법과 수치를 재현하지 않음"},
 {"id":"S13","title":"3GPP TR 36 777 Enhanced LTE support for aerial vehicles","url":"https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3231","location":"공식 규격 정보","use":"교수님이 예시로 든 채널 모델 후보","evidence":"검토 후보. 본 코드에는 해당 표의 계수를 구현하지 않았음"},
 {"id":"S14","title":"capstone simulation maps team repository","url":"https://github.com/ytw1227/capstone-simulation-maps/tree/bf86e773d91f7bdd3567c7a4f6ebdd7be379335f","location":"README docs simulation handoff와 scene local json","use":"실제 건물 폴리곤·높이·비행 금지 버퍼와 데이터 출처","evidence":"제공 저장소의 고정 커밋. 추정 높이는 실측 높이와 구분"}
]


# 지표 사전을 만들어 CSV와 화면의 한국어 이름·단위·정의에 사용합니다.
def catalog():
    rows=[]
    # 한 지표의 키·한국어명·단위·선호방향·근거·정의를 사전에 추가합니다.
    def add(key,label,unit,direction,definition,source="자체 구현",origin="구현 정의"):
        rows.append({"key":key,"label_ko":label,"unit":unit,"direction":direction,
                     "definition":definition,"source":source,"origin":origin})
    graph=[
      ("fleet_connected","전체 연결 유지율","fraction","max","초기 전체 N개의 정점이 한 연결 성분인 시간 비율"),
      ("search_connected","수색 드론 전체 연결 유지율","fraction","max","수색 드론 모두가 하나의 성분에 속한 시간 비율. 전용 중계 경유 허용"),
      ("search_pair_reachability","수색 쌍 도달 가능 비율","fraction","max","연결된 수색 쌍 수 / Ns(Ns-1)/2 의 시간 평균"),
      ("components","연결 성분 수","count","min","가용 링크 그래프의 연결 성분 수 평균. 이탈한 정점도 초기 N 분모에 유지"),
      ("largest_component_fraction","최대 성분 비율","fraction","max","가장 큰 성분의 정점 수 / 초기 N"),
      ("mean_degree","평균 이웃 수","count","context","정점별 degree의 평균. 큰 값만으로 수색 우수성을 판단하지 않음"),
      ("isolated_fraction","고립 드론 비율","fraction","min","degree가 0인 정점 수 / 초기 N"),
      ("edge_density","링크 밀도","fraction","context","가용 무방향 링크 수 / N(N-1)/2"),
      ("normalized_lambda2","정규화 라플라시안 두 번째 고유값","dimensionless","context","Lnorm=D^(-1/2)(D-A)D^(-1/2), 고립점은 대각 0. 두 번째 고유값. 0~1 점수가 아님"),
      ("graph_efficiency","홉 기반 그래프 효율","fraction","max","모든 정점 쌍의 1/최단 홉 수 평균. 연결 불가능한 쌍은 0"),
      ("reachable_search_hops","도달 가능한 수색 쌍 평균 홉 수","count","context","도달 가능한 쌍에 한정한 최단 홉 수. 단절한 쌍을 숨길 수 있어 도달률과 함께 해석"),
      ("formation_rmse_m","평균 대형 오차","m","min","수색군의 중심 이동을 제거하고 목표 슬롯과 실제 위치 차이의 RMS를 프레임별 평균"),
      ("los_fraction","드론 쌍 LoS 비율","fraction","context","드론 높이에서 직선이 건물 지붕에 막히지 않은 쌍 비율. 수색용 지면 가시성과 다름"),
      ("sinr_p05_db","프레임별 링크 SINR 하위 5백분위 평균","dB","context","각 드론 쌍의 양방향 SINR 중 낮은 값의 5백분위를 시간 평균"),
      ("sinr_mean_db","방향별 평균 SINR","dB","context","i<j 방향의 dB SINR 시간 평균. 링크 가용성은 양방향을 판정"),
      ("rx_mean_dbm","평균 수신 전력","dBm","context","i<j 방향 수신 전력 dBm의 시간 평균"),
      ("path_loss_mean_db","평균 전파 손실","dB","context","자유공간 손실 + 선택한 장애물 추가 손실 평균"),
      ("single_node_removal_survival","단일 정점 제거 후 수색 연결 유지 비율","fraction","max","각 정점을 가상 제거했을 때 남은 수색 정점들이 연결되는 제거 사례 비율의 시간 평균. 실제 복구 정책 아님"),
      ("relay_removal_survival","전용 중계 제거 후 연결 유지 비율","fraction","max","전용 중계를 하나씩 가상 제거한 정적 점검. 전용 중계가 없으면 값 없음"),
      ("queue_packets","평균 대기 패킷 수","count","min","운동 프레임 끝에서 모든 정점 큐 길이의 합을 평균")
    ]
    for key,label,unit,direction,definition in graph:
        source="S09" if key=="normalized_lambda2" else "S07" if key=="graph_efficiency" else "자체 구현"
        add(key+"_mean",label,unit,direction,definition,source,"수식 참고 후 적용" if source!="자체 구현" else "구현 정의")
    extras=[
      ("search_outage_total_s","누적 수색 단절 시간","s","min","전체 수색 연결이 거짓인 프레임 수 × 운동 dt"),
      ("search_outage_events","수색 단절 구간 수","count","min","연속 단절 프레임을 하나의 구간으로 묶은 수"),
      ("search_outage_max_s","최장 수색 단절 시간","s","min","가장 긴 연속 단절 구간의 길이"),
      ("formation_rmse_p95_m","대형 오차 95백분위","m","min","프레임별 수색 대형 RMS 오차의 95백분위"),
      ("observed_coverage_final","누적 관측 비율","fraction","max","어떤 수색 드론이든 관측한 실외 목표 셀 합집합 / 고정된 실외 셀 수"),
      ("shared_coverage_final","수색군 공통 공유 비율","fraction","max","모든 수색 드론의 지식 집합 교집합 / 고정된 실외 셀 수. 실제 전달된 페이로드만 지식에 추가"),
      ("coverage_jain_final","관측량 Jain 공정성","fraction","context","드론별 관측 셀 수에 Jain 식 적용. 관측량 자체와 함께 확인"),
      ("coverage_overlap_final","관측 중복 비율","fraction","min","1 - 관측 합집합 셀 수 / 드론별 관측 셀 수 합"),
      ("total_travel_m","전체 이동량","m","min","프레임별 실제 선형 이동 길이를 전체 드론에 합산. 에너지 J가 아님"),
      ("search_travel_m","수색군 이동량","m","context","수색 드론 이동 길이 합"),
      ("relay_travel_m","전용 중계 이동량","m","context","전용 중계 이동 길이 합"),
      ("minimum_swept_separation_m","이동 구간 최소 드론 거리","m","constraint","각 프레임의 동시 선형 이동 구간 전체에서의 최소 상대 거리. 프레임 끝만 검사하지 않음"),
      ("stationary_agent_frames","미이동 드론 프레임 수","agent frames","context","활성 드론이 이동하지 않은 프레임 수 합. 자연 대기와 충돌 대기를 모두 포함"),
      ("boundary_clamp_agent_frames","목표 위치 경계 보정 수","agent frames","min","목표 슬롯이 AOI 격자 경계를 넘어 보정된 드론 프레임 수"),
      ("link_changes","링크 상태 변경 수","count","context","인접 프레임 사이 가용 링크의 생성·소멸 수 합"),
      ("link_changes_per_s","초당 링크 상태 변경 수","1/s","context","링크 변경 수 / 실험 생성 구간 길이"),
      ("obstacle_violations","비행 장애물 통과 수","count","constraint","유효 실행에서는 0. 통과 구간이 생성되면 실행을 오류로 중단"),
      ("flight_free_fraction","비행 가능 격자 비율","fraction","context","비행 금지 영역 밖 격자 셀 수 / 전체 격자 셀 수"),
      ("retained_free_cell_fraction","선택한 비행 성분 비율","fraction","context","최대 비행 성분의 격자 셀 수 / 모든 자유 셀 수. 작은 고립 성분 제외를 공개"),
      ("jammer_active_fraction","간섭 활성 시간 비율","fraction","context","실제로 간섭원을 켠 프레임 비율")
    ]
    for x in extras:add(*x,"S08" if x[0]=="coverage_jain_final" else "자체 구현","수식 참고 후 적용" if x[0]=="coverage_jain_final" else "구현 정의")
    packet=[
      ("packets_generated","생성 패킷 수","count","context","0<=생성 시각<T인 종단 간 패킷 수"),
      ("packets_delivered","전달 패킷 수","count","max","TTL 안에 원래 수색 목적지까지 도착한 패킷 수"),
      ("packets_pending","종료 후 미처리 패킷 수","count","constraint","생성 중단 후 TTL까지 배출한 뒤 남은 수. 정상 결과는 0"),
      ("pdr","종단 간 전달률","fraction","max","TTL 안에 전달된 패킷 / 생성 패킷. 배출 구간 전달도 생성 코호트에 포함"),
      ("timely_pdr","적시 전달률","fraction","max","생성 후 적시 기준 이내 도착한 패킷 / 생성 패킷. 기준 0.25초는 자체 설정"),
      ("goodput_kbps","유효 처리량","kbps","max","실험 구간 T 안에 도착한 페이로드 bit / T / 1000. 헤더·ACK 제외"),
      ("delay_mean_s","전달 패킷 평균 지연","s","min","도착 시각 - 생성 시각. 성공 전달에 조건부인 값"),
      ("delay_p95_s","전달 패킷 지연 95백분위","s","min","전달된 패킷 지연의 95백분위. 유실은 지연 0으로 넣지 않음"),
      ("delay_max_s","전달 패킷 최대 지연","s","min","전달된 패킷 중 가장 긴 지연"),
      ("delivered_hops_mean","전달 패킷 평균 홉 수","count","context","성공 패킷이 도착할 때까지 성공한 홉 수 평균"),
      ("dedicated_relay_hops_mean","전용 중계 사용 홉 수","count","context","성공 패킷 경로 중 송신자가 전용 중계였던 홉 수 평균"),
      ("flow_pdr_jain","흐름별 PDR Jain 공정성","fraction","context","각 수색 src dst 흐름의 PDR를 Jain 식에 입력. 모두 0이면 값 없음"),
      ("flow_pdr_min","흐름별 최저 PDR","fraction","max","생성 패킷이 있는 흐름 중 가장 낮은 전달률"),
      ("flow_pdr_std","흐름별 PDR 표준편차","fraction","min","수색 src dst 흐름 전달률의 모집단 표준편차"),
      ("tx_attempts","홉 전송 시도 수","count","context","모든 데이터·ACK 묶음 홉 시도 수. 성공과 실패 포함"),
      ("hop_successes","성공 홉 수","count","context","성공한 단일 홉 시도 수. 종단 간 전달 수와 다름"),
      ("hop_success_rate","홉 시도 성공률","fraction","max","성공 홉 / 홉 시도. 송신하지 못한 no route 슬롯은 분모에 없음"),
      ("retries","재전송 시도 수","count","min","같은 홉에서 첫 시도 이후에 이루어진 추가 시도 수"),
      ("tx_per_delivered","전달 1개당 전송 시도 수","count","min","전체 홉 시도 / 종단 간 성공 전달. 성공 0이면 값 없음"),
      ("wire_traffic_kbytes","모형상 전송량","kB","context","홉 시도 × 페이로드·헤더·ACK byte 합. 실패 시도에도 전체 ACK 예산을 포함한 추상 값"),
      ("queue_max_packets","최대 정점 큐 길이","count","min","어느 정점에서든 관측한 가장 큰 큐 길이"),
      ("no_route_slots","경로 없이 대기한 기회 수","slots","min","큐가 있지만 목적지 경로가 없어 보내지 못한 송신 기회 수"),
      ("route_updates","경로 스냅샷 갱신 수","count","context","정해진 주기마다 경로를 다시 계산한 수. 배출 구간 포함. 제어 패킷 전송 수 아님"),
      ("drop_expired","패킷 수명 초과 유실","count","min","생성 후 TTL을 넘겨 제거한 패킷 수"),
      ("drop_retry_limit","재시도 한도 유실","count","min","한 홉에서 최대 시도 수에 도달해 제거한 패킷 수"),
      ("drop_queue_overflow","큐 한도 유실","count","min","수신 또는 생성 시 해당 큐의 한도를 넘은 패킷 수"),
      ("drop_node_failure","드론 이탈 유실","count","min","이탈한 정점의 큐에 있던 패킷 수"),
      ("aoi_mean_s","샘플 평균 정보 연령","s","min","프레임 끝에서 현재 시각 - 최신 수신 상태 패킷 생성 시각을 흐름별·시간별 평균. 초기 timestamp는 0"),
      ("aoi_peak_s","샘플 최대 정보 연령","s","min","모든 수색 흐름과 프레임 끝에서의 최대 AoI. 관측 셀 자체의 나이가 아님"),
      ("never_received_flow_fraction","한 번도 수신하지 못한 흐름 비율","fraction","min","T까지 상태 패킷을 한 번도 받지 못한 수색 src dst 흐름 비율")
    ]
    for x in packet:
        k=x[0];source="S05" if k in ("pdr","packets_delivered","drop_expired") else "S06" if k.startswith("delay") else "S08" if k=="flow_pdr_jain" else "S10" if k.startswith("aoi") else "자체 구현"
        add(*x,source,"수식 또는 측정 원칙 참고 후 적용" if source!="자체 구현" else "구현 정의")
    for key,label,definition in [
      ("nearest_neighbor_distance_mean_m","전체 드론 최근접 거리 평균","활성 드론별 가장 가까운 다른 드론의 거리 전체 시간 평균"),
      ("nearest_neighbor_distance_p05_m","실제 최근접 거리 5백분위","활성 드론별 최근접 거리를 모든 프레임에 모은 5백분위"),
      ("maximum_pair_distance_mean_m","전체 드론 최대 쌍 거리 평균","프레임별 가장 먼 드론 쌍 거리의 시간 평균"),
      ("search_nearest_neighbor_mean_m","수색 드론 최근접 거리 평균","각 수색 드론과 가장 가까운 다른 수색 드론의 거리 평균"),
      ("search_pair_distance_mean_m","수색 쌍 실제 거리 평균","모든 수색 드론 쌍 거리의 시간 평균"),
      ("search_spread_radius_mean_m","수색군 실제 배치 반경","수색군 중심에서 가장 먼 수색 드론까지 거리의 시간 평균")
    ]:add(key,label,"m","context",definition)
    return rows
