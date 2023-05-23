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
} from "recharts";
import styles from "./page.module.css";
import { useEffect,useState } from "react";
import axios from "axios";
import internal from "stream";

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

export default function Table() {

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

  const [state, setState] = useState({
    opacity: {
      uv: 1,
      pv: 1,
    },
  });
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

  if(data != undefined){
  return (
    <div style={{ width: "100%" }} className={styles.container}>
      <p>Wrapper com perceptron em Python com a biblioteca Sklearn</p>
      <p>A precisão do wrapper com AG foi de: {100 * Number(data?.wrapper_AG_acurracy.toPrecision(4))}%</p>
      <p>Foram utilizados&nbsp; 
       {
         data?.wrapper_AG_solution.filter(
          (value,index)=>{if(value == 1) 
            return index
          }).length+1
      }&nbsp;parametros de {data.wrapper_AG_solution.length}, sendo eles:
     </p>
     [{
      
      data?.wrapper_AG_solution.map(
        (value,index)=>{if(value == 1) 
          return index + ','
        })
        
    }]
      <p>O tempo necessário para concluir a busca foi de: {data?.elapsed_time_AG.toPrecision(3)} segundos</p>

      <br/>

      <p>A precisão do wrapper com Hill Climbing foi de: {100 * Number(data?.wrapper_hillClimbing_acurracy.toPrecision(4))}%</p>
      <p>Foram utilizados&nbsp;
       {
         data?.wrapper_hillClimbing_solution.filter(
          (value,index)=>{if(value == 1) 
            return index
          }).length
      }&nbsp;parametros de {data.wrapper_hillClimbing_solution.length}, sendo eles:</p>
      [{data?.wrapper_hillClimbing_solution.map(
        (value,index)=>{if(value == 1) return index + ','})}]
      <p>O tempo necessário para concluir a busca foi de: {data?.elapsed_time_hill.toPrecision(3)} segundos</p>
    </div>
    );
  }else{
    return <div>
      <p>Carregando</p>
    </div>
  }
}
