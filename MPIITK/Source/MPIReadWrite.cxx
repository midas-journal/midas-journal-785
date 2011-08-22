#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include "itkVector.h"

#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageIORegion.h"

#include "itkImageRegionSplitter.h"

#define TAG_PIECE   0

int main( int argc, char *argv[] )
{

  const char *       in_file_name    =       argv[1];
  const char *       out_file_name   =       argv[2];

  const unsigned int Dimension = 3;
  const unsigned int Channels  = 3;
  typedef itk::Vector <float, Channels >           VectorPixelType;
  typedef itk::Image< VectorPixelType, Dimension > VectorImageType;
  typedef VectorImageType::RegionType              RegionType;
  typedef VectorImageType::IndexType               IndexType;
  typedef VectorImageType::SizeType                SizeType;
  typedef VectorImageType::PointType               PointType;
  typedef VectorImageType::SpacingType             SpacingType;
  typedef itk::Image< float, Dimension >           ScalarImageType;
  
  int mpi_rank;
  int mpi_size;

  // Initialise MPI
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_size ); 

  typedef itk::PasteImageFilter < 
    VectorImageType, VectorImageType, VectorImageType > PasteType;

  // Reader
  typedef itk::ImageFileReader< VectorImageType > FileReaderType;
  FileReaderType::Pointer reader = FileReaderType::New();
  reader->SetFileName( in_file_name );
  try
    {
    reader->UpdateOutputInformation();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught while updating reader information!" << std::endl;
    std::cerr << excep << std::endl;
    }

  VectorImageType::Pointer input_image = reader->GetOutput();
  PointType   input_origin  = input_image->GetOrigin();
  RegionType  input_region  = input_image->GetLargestPossibleRegion();
  IndexType   input_index   = input_region.GetIndex();
  SizeType    input_size    = input_region.GetSize();
  SpacingType input_spacing = input_image->GetSpacing();

  // Region Splitter
  typedef itk::ImageRegionSplitter< Dimension > SplitterType;
  SplitterType::Pointer splitter = SplitterType::New();

  // Get The splits
  std::vector < RegionType > split_regions( mpi_size );
  for ( int split = 0; split < mpi_size; ++split )
    {
    split_regions[ split ] 
      = splitter->GetSplit( split, mpi_size, input_region );
    } // end for split

  // Extractor Type
  typedef itk::ExtractImageFilter< 
    VectorImageType, VectorImageType > ExtractorType;
  ExtractorType::Pointer extractor = ExtractorType::New();
  extractor->SetInput( reader->GetOutput() );
  extractor->SetExtractionRegion( split_regions[ mpi_rank ] );

  try
    {
    extractor->Update();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught while updating extractor!" << std::endl;
    std::cerr << excep << std::endl;
    }

  // Each process, other than 0, sends it's image to process 0.
  if ( mpi_rank != 0 )
    {
    MPI_Ssend( 
      extractor->GetOutput()->GetBufferPointer(),
      split_regions[ mpi_rank ].GetNumberOfPixels() * Channels,
      MPI_FLOAT, 0, TAG_PIECE, MPI_COMM_WORLD );
    }
  else
    {
    for ( int split = 0; split < mpi_size; ++split )
      { 
      // Get the receive image region.
      RegionType recv_region = split_regions[ split ];
      IndexType write_index = recv_region.GetIndex();
      SizeType  write_size  = recv_region.GetSize();

      // Make the receive image.
      VectorImageType::Pointer recv_image = VectorImageType::New();
      recv_image->SetOrigin( input_origin );
      recv_image->SetRegions( recv_region );
      if ( split != 0 )
        {
        recv_image->Allocate();

        // Receive the image.
        MPI_Recv( recv_image->GetBufferPointer(), 
          recv_region.GetNumberOfPixels() * Channels, 
          MPI_FLOAT, split, TAG_PIECE, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }

      // Paste
      PasteType::Pointer paste = PasteType::New();
      paste->SetDestinationImage( reader->GetOutput() );
      if ( split == 0 )
        {
        paste->SetSourceImage( extractor->GetOutput() );
        }
      else
        {
        paste->SetSourceImage( recv_image );
        }
      paste->SetDestinationIndex( write_index );
      paste->SetSourceRegion( recv_region );
        
      // Writer
      typedef itk::ImageFileWriter< VectorImageType > FileWriterType;
      FileWriterType::Pointer writer = FileWriterType::New();
      writer->SetFileName( out_file_name );
      writer->SetInput( paste->GetOutput() );

      itk::ImageIORegion write_region( Dimension );
      for ( unsigned int dim = 0; dim < Dimension; ++dim )
        {
        write_region.SetIndex( dim, write_index[ dim ] );
        write_region.SetSize(  dim, write_size[  dim ] );
        }
      writer->SetIORegion( write_region );

      try
        {
         std::cerr << "Piece " << split << 
           " updating writer." << std::endl;
        writer->Update();
        }
      catch( itk::ExceptionObject & excep )
        {
        std::cerr << "Exception caught !" << std::endl;
        std::cerr << excep << std::endl;
        }
      } // end split loop
    } // end rank 0
   
  MPI_Finalize();


  return EXIT_SUCCESS;
}
