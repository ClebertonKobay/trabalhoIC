import Image from "next/image";
import styles from "./page.module.css";
import Graphic from "@/components/graphic/page";

export default function Home() {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Trabalho Primeiro Bimestre</h1>
      <p className={styles.description}>Descrição</p>
      <div className={styles.graphicsArea}>
        <Graphic />
      </div>
      <div className={styles.footer}>
        <h3>
          Trabalho desenvolvido para a disciplina de Inteligência Computacional
        </h3>
        <div className={styles.authors}>
          <p>Cleberton Kobay dos Santos</p>
          <p>Fernanda Ramos Paulino</p>
          <p>Maria Eduarda Grocholski</p>
          <p>Mateus José da Silva Gomes</p>
          <p>Matheus Bueno Bartkevicius</p>
        </div>
        <div className={styles.local}>
          <p>UEPG | 2023</p>
        </div>
      </div>
    </div>
  );
}
