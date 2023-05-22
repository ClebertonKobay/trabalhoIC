"use client";
import {
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Bar,
  LineChart,
  ResponsiveContainer,
  Line,
  Text,
  Label,
} from "recharts";
import styles from "./page.module.css";
import { useEffect,useState } from "react";
import axios from "axios";
import { NamedTupleMember } from "typescript";

const teste = [
  { name: "a", uv: 12 },
  { name: "b", uv: 5 },
  { name: "c", uv: 5 },
  { name: "d", uv: 8 },
  { name: "e", uv: 20 },
  { name: "f", uv: 2 },
];

interface Data{
  wrapper_AG_solution : Array<number>,
  wrapper_AG_acurracy : number,
  elapsed_time_AG :number ,
  wrapper_hillClimbing_solution : Array<number>,
  wrapper_hillClimbing_acurracy : number,
  elapsed_time_hill : number,
  All_accuracy_AG: Array<number>,
  All_accuracy_Hill: Array<number>,
}


export default function Graphic() {
  const [state, setState] = useState({
    opacity: {
      uv: 1,
      pv: 1,
    },
  });

  const [data, setData] = useState<Data>();

  const api = axios.create({
   baseURL: 'http://localhost:5000'
  })

  useEffect(() => {
  const fetchData = async () => {
    try {
      await api.get("predict").then(response=>{
        const dadosJson: Data = response.data;
        setData(dadosJson);
      });
      // Armazena os dados na state "data"
    } catch (error) {
      console.error("Erro ao obter os dados:", error);
    }
  };
  
  fetchData(); // Chama a função de requisição ao carregar os dados
}, [api]);

  const dataToGrafic = (dataArray1: Array<number>,dataArray2: Array<number>):{"AG": number,"Hill":number,"index":number}[]=> {
    const dataGrafic = dataArray1.map((element,index) => {
      return {"AG": element,"Hill": dataArray2[index],'index': index};
    });
    console.log(dataGrafic);
    return dataGrafic;
  };

  const handleMouseEnter = (o: any) => {
    const { dataKey } = o;
    const { opacity } = state;

    setState({
      opacity: { ...opacity, [dataKey]: 0.5 },
    });
  };

  const handleMouseLeave = (o: any) => {
    const { dataKey } = o;
    const { opacity } = state;

    setState({
      opacity: { ...opacity, [dataKey]: 1 },
    });
  };
  if (data != undefined){
  return (
    <div style={{ width: "90vw" }} className={styles.container}>
      <Text>Gráfico de comparação entre wrapper com AG e Hill-Climbing</Text>
      <ResponsiveContainer  height={300}>
        <LineChart  data={dataToGrafic(data.All_accuracy_AG,data.All_accuracy_Hill)}>
          <XAxis  />
          <YAxis dataKey="AG" label={{ value: 'Acurácia', angle: -90, position: 'insideLeft', textAnchor: 'middle' }} />
          <Tooltip />
          <CartesianGrid stroke="#eee" strokeDasharray="5 5"/>
          <Legend />
          <Line type="monotone" onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave} dataKey="AG" stroke="#FF0000" />
          <Line type="monotone" dataKey="Hill" stroke="#0000FF" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
  }else{
    return <div>
    </div>
  }
}