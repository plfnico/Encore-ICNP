#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/log.h"
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

using namespace ns3;

struct FLowInfo {
    uint32_t flowIndex;
    uint32_t leafIndex; 
    double startTime;
    uint64_t flowSize; 
    double completionTime = 0;
    uint128_t receivedBytes = 0;
};
uint32_t num_flows;
std::vector<FLowInfo> flows;
std::vector<uint32_t> totalPackets;
std::vector<uint32_t> dropPackets;
std::ofstream fct_out, drop_out, tp_out, qlen_out;
uint32_t throughPut = 0;
const uint32_t initialPort = 5000;
double curTime = 0;
double queueLen = 0;

void PrintSimulationTime() {
    // std::cout << "Current simulation time: " << Simulator::Now().GetSeconds() << "s" << std::endl;
    tp_out << Simulator::Now().GetSeconds() << "," << throughPut << std::endl;
    qlen_out << Simulator::Now().GetSeconds() << "," << queueLen << std::endl;
    throughPut = 0;
    queueLen = 0;
    Simulator::Schedule(Seconds(1), &PrintSimulationTime);
}


void ReceivedPacket(uint32_t flowId, Ptr<const Packet> packet, const Address& address) {
    uint32_t leafIndex = flows[flowId].leafIndex;
    totalPackets[leafIndex]++;
    flows[flowId].receivedBytes += packet->GetSize();
    throughPut += packet->GetSize();
    if (flows[flowId].receivedBytes >= flows[flowId].flowSize) {
        double endTime = Simulator::Now().GetSeconds();
        flows[flowId].completionTime = endTime - flows[flowId].startTime;
        fct_out << flowId << "," << flows[flowId].leafIndex << "," << flows[flowId].flowSize << "," << flows[flowId].startTime << "," << endTime << "," << flows[flowId].completionTime << std::endl;
        fct_out.flush();
    }
}


bool ReadFlowInfo(const std::string& flowFile, uint32_t& numFlows, std::vector<FLowInfo>& streamInfos){
    std::ifstream file(flowFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << flowFile << std::endl;
        return false;
    }
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> numFlows;
    }
    uint32_t flowCount = 0;
    for (uint32_t i = 0; i < numFlows; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        FLowInfo info;
        iss >> info.leafIndex >> info.startTime >> info.flowSize;
        info.flowIndex = flowCount;
        streamInfos.push_back(info);
        flowCount++;
    }
    file.close();
    if (flowCount != numFlows) {
        std::cerr << "Number of flows in the file does not match the number of flows specified in the first line" << std::endl;
        return false;
    }
    return true;
}


void DropPacket(Ptr<const QueueDiscItem> item) {
    Ptr<const Packet> packet = item->GetPacket();
    TcpHeader tcpHeader;
    packet->PeekHeader(tcpHeader);
    uint16_t dstPort = tcpHeader.GetDestinationPort();
    uint32_t flowId = dstPort - initialPort;
    if(flowId >= num_flows) {
        std::cerr << "Flow ID is greater than the number of flows" << std::endl;
    }
    uint32_t leaf = flows[flowId].leafIndex;
    dropPackets[leaf]++;
    totalPackets[leaf]++;
}


void BytesInQueueTrace(uint32_t oldValue, uint32_t newValue) {
    queueLen += oldValue * (Simulator::Now().GetSeconds() - curTime);
    curTime = Simulator::Now().GetSeconds();
}


int
main(int argc, char* argv[])
{
    clock_t begint, endt;
	begint = clock();

    // read configuration file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>" << std::endl;
        return 1;
    }

    std::string configFilePath = argv[1];
    std::ifstream conf(configFilePath);
    if (!conf.is_open()) {
        std::cerr << "Failed to open file: " << configFilePath << std::endl;
        return 1;
    }
    std::cout << "Reading configuration from " << configFilePath << std::endl;

    // read configuration    
    uint32_t numLeaf;
    std::string flowFile;
    std::string fctFile, dropFile, tpFile, qlenFile;
    uint32_t redMinTh = 1000;
    uint32_t redMaxTh = 100000;
    double redPmax = 0.1;

    while (!conf.eof()){
        std::string key;
        conf >> key;
        if (key.compare("NUM_LEAF") == 0){
            conf >> numLeaf;
            // std::cout << "Number of leaf nodes: " << numLeaf << std::endl;
        }
        else if (key.compare("FLOW_FILE") == 0){
            conf >> flowFile;
            // std::cout << "Flow file: " << flowFile << std::endl;
        }
        else if (key.compare("RED_MIN_TH") == 0){
            conf >> redMinTh;
            // std::cout << "RED min threshold: " << redMinTh << std::endl;
        }
        else if (key.compare("RED_MAX_TH") == 0){
            conf >> redMaxTh;
            // std::cout << "RED max threshold: " << redMaxTh << std::endl;
        }
        else if (key.compare("RED_PMAX") == 0){
            conf >> redPmax;
            // std::cout << "RED pmax: " << redPmax << std::endl;
        }
        else if (key.compare("FCT_FILE") == 0){
            conf >> fctFile;
            // std::cout << "FCT file: " << fctFile << std::endl;
        }
        else if (key.compare("DROP_FILE") == 0){
            conf >> dropFile;
            // std::cout << "Drop file: " << dropFile << std::endl;
        }
        else if (key.compare("THROUGHPUT_FILE") == 0){
            conf >> tpFile;
            // std::cout << "Throughput file: " << tpFile << std::endl;
        }
        else if (key.compare("QLEN_FILE") == 0){
            conf >> qlenFile;
            // std::cout << "Queue length file: " << qlenFile << std::endl;
        }
        else {
            std::cerr << "Unknown key: " << key << std::endl;
            return 1;
        }
    }
    conf.close();

    // static configuration
    uint32_t pktSize = 512;
    uint32_t port = initialPort;
    uint32_t bufferSize = 4000000;
    std::string AccessLinkBw = "200Mbps";
    std::string AccessLinkDelay = "1ms";
    std::string bottleNeckLinkBw = "200Mbps";
    std::string bottleNeckLinkDelay = "20ms";
    std::string transportProtocol = "ns3::TcpCubic";

    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName(transportProtocol)));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::BulkSendApplication::SendSize", UintegerValue(pktSize));
    Config::SetDefault("ns3::RedQueueDisc::LinkBandwidth", StringValue(bottleNeckLinkBw));
    Config::SetDefault("ns3::RedQueueDisc::LinkDelay", StringValue(bottleNeckLinkDelay));
    Config::SetDefault("ns3::RedQueueDisc::MinTh", DoubleValue(redMinTh));
    Config::SetDefault("ns3::RedQueueDisc::MaxTh", DoubleValue(redMaxTh));
    Config::SetDefault("ns3::RedQueueDisc::LInterm", DoubleValue(1 / redPmax));
    Config::SetDefault("ns3::RedQueueDisc::Gentle", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::QW", DoubleValue(1.0));
    Config::SetDefault(
            "ns3::RedQueueDisc::MaxSize",
            QueueSizeValue(QueueSize(QueueSizeUnit::BYTES, bufferSize)));

    totalPackets = std::vector<uint32_t>(numLeaf, 0);
    dropPackets = std::vector<uint32_t>(numLeaf, 0);

    // Create the point-to-point link helpers
    PointToPointHelper bottleNeckLink;
    bottleNeckLink.SetDeviceAttribute("DataRate", StringValue(bottleNeckLinkBw));
    bottleNeckLink.SetChannelAttribute("Delay", StringValue(bottleNeckLinkDelay));

    PointToPointHelper pointToPointLeaf;
    pointToPointLeaf.SetDeviceAttribute("DataRate", StringValue(AccessLinkBw));
    pointToPointLeaf.SetChannelAttribute("Delay", StringValue(AccessLinkDelay));

    PointToPointDumbbellHelper d(numLeaf, pointToPointLeaf, numLeaf, pointToPointLeaf, bottleNeckLink);

    // Install Stack
    InternetStackHelper stack;
    for (uint32_t i = 0; i < d.LeftCount(); ++i)
    {
        stack.Install(d.GetLeft(i));
    }
    for (uint32_t i = 0; i < d.RightCount(); ++i)
    {
        stack.Install(d.GetRight(i));
    }
    stack.Install(d.GetLeft());
    stack.Install(d.GetRight());
    TrafficControlHelper tchBottleneck;
    tchBottleneck.SetRootQueueDisc("ns3::RedQueueDisc");
    QueueDiscContainer qdiscs;
    qdiscs = tchBottleneck.Install(d.GetRight()->GetDevice(0));
    Ptr<QueueDisc> q = qdiscs.Get(0);
    q->TraceConnectWithoutContext("Drop", MakeCallback(&DropPacket));
    q->TraceConnectWithoutContext("BytesInQueue", MakeCallback(&BytesInQueueTrace));

    // Assign IP Addresses
    d.AssignIpv4Addresses(Ipv4AddressHelper("10.1.1.0", "255.255.255.0"),
                          Ipv4AddressHelper("10.2.1.0", "255.255.255.0"),
                          Ipv4AddressHelper("10.3.1.0", "255.255.255.0"));
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    bool readFLowFlag = ReadFlowInfo(flowFile, num_flows, flows);
    if (!readFLowFlag) {
        std::cerr << "Failed to read flow file" << std::endl;
        return 1;
    }

    fct_out.open(fctFile, std::ios::out);
    if (!fct_out.is_open()) {
        std::cerr << "Failed to open file: " << fctFile << std::endl;
        return 1;
    }
    fct_out << "FlowID,Leaf,Size,StartTime,EndTime,CompletionTime" << std::endl;

    drop_out.open(dropFile, std::ios::out);
    if (!drop_out.is_open()) {
        std::cerr << "Failed to open file: " << dropFile << std::endl;
        return 1;
    }
    drop_out << "Leaf,TotalPacket,DropPacket" << std::endl;

    tp_out.open(tpFile, std::ios::out);
    if (!tp_out.is_open()) {
        std::cerr << "Failed to open file: " << tpFile << std::endl;
        return 1;
    }
    tp_out << "Time,Throughput" << std::endl;

    qlen_out.open(qlenFile, std::ios::out);
    if (!qlen_out.is_open()) {
        std::cerr << "Failed to open file: " << qlenFile << std::endl;
        return 1;
    }
    qlen_out << "Time,QueueLength" << std::endl;

    for (auto it: flows) {
        uint32_t flowID = it.flowIndex;
        uint32_t leafIndex = it.leafIndex;
        double startTime = it.startTime;
        uint64_t flowSize = it.flowSize;
        BulkSendHelper clientHelper("ns3::TcpSocketFactory", Address());
        clientHelper.SetAttribute("MaxBytes", UintegerValue(flowSize));
        
        InetSocketAddress remoteAddr(d.GetLeftIpv4Address(leafIndex), port);
        clientHelper.SetAttribute("Remote", AddressValue(remoteAddr));

        ApplicationContainer clientApp = clientHelper.Install(d.GetRight(leafIndex));
        clientApp.Start(Seconds(startTime));
        clientApp.Stop(Seconds(startTime + 120));

        Address sinkLocalAddress(InetSocketAddress(d.GetLeftIpv4Address(leafIndex), port));
        PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
        ApplicationContainer sinkApp = packetSinkHelper.Install(d.GetLeft(leafIndex));
        sinkApp.Start(Seconds(startTime));
        sinkApp.Stop(Seconds(startTime + 120));
        Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApp.Get(0));
        sink->TraceConnectWithoutContext("Rx", MakeBoundCallback(&ReceivedPacket, flowID));
        port++;
    }
    Simulator::Schedule(Seconds(1), &PrintSimulationTime);

    Simulator::Stop(Seconds(180));
    Simulator::Run();
    std::cout << "Destroying the simulation " << std::endl;
    Simulator::Destroy();

    for(uint32_t i = 0; i < numLeaf; i++){
        drop_out << i << "," << totalPackets[i] << "," << dropPackets[i] << std::endl;
    }
    drop_out.flush();
    drop_out.close();
    fct_out.flush();
    fct_out.close();
    tp_out.flush();
    tp_out.close();
    qlen_out.flush();
    qlen_out.close();
    endt = clock();
    uint32_t elapsed_mins = (endt - begint) / CLOCKS_PER_SEC / 60;
    uint32_t elapsed_secs = (endt - begint) / CLOCKS_PER_SEC % 60;
    std::cout << "Simulation time " << elapsed_mins << "m " << elapsed_secs << "s" << std::endl;

    return 0;
}
