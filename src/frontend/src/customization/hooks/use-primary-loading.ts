import { UseRequestProcessor } from "@/controllers/API/services/request-processor";
import { useQueryFunctionType } from "@/types/api";

export const usePrimaryLoading: useQueryFunctionType<undefined, null> = (
  options,
) => {
  const { query } = UseRequestProcessor();

  const getPrimaryLoadingFn = async () => {
    return null;
  };

  const queryResult = query(
    ["usePrimaryLoading"],
    getPrimaryLoadingFn,
    options,
  );

  return queryResult;
};